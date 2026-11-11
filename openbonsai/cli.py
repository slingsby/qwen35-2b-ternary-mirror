"""``openbonsai`` command line.

    openbonsai run --model Qwen/Qwen3.5-9B --config examples/qwen3_5-9b.yaml --out runs/my-model
        = traces -> calibrate -> quantize -> endpoints -> [refine] -> distill -> export -> eval
        (refine is off by default: refine.enabled=true; docs/refine.md)

    openbonsai <stage> --run runs/my-model [--set section.key=value ...] [--force]
        stages: traces calibrate quantize endpoints refine distill export eval

    openbonsai export --run runs/my-model [--out X.gguf|DIR] [--name NAME] [--verify | --no-verify]
        a PTQ1_0 GGUF, or a latent checkpoint folder when the model has no GGUF layout (export.format);
        DIR is the checkpoint folder itself (an existing one is replaced only if it is a latent checkpoint)
    openbonsai eval --run runs/my-model --from bf16|quantize|endpoints|refine|distill|PATH --suite dev,extended,compare,teacher
        PATH: state.safetensors, legacy .pt[,.pt], or a PTQ1_0 .gguf (ours or the official
        Ternary Bonsai; --run defaults to <gguf>.eval and --model gives the config and tokenizer)
    openbonsai eval --run runs/my-model --from distill --suite gsm8k,humaneval --set 'eval.reasoning.thinking=[off,on]'
        greedy chat generation on frozen GSM8K/HumanEval TEST subsets (docs/evaluation.md)
    openbonsai inspect PATH            summary of a state file, GGUF, latent checkpoint folder or run directory

Errors print one line (``openbonsai: error: ...``) and exit 2; ``--debug`` (or
OPENBONSAI_DEBUG=1) shows the traceback instead. A stage that fails also writes
its traceback to the run's ``logs/<stage>.log``.

A stage whose output exists with the same config hash is skipped; ``--force``
reruns it from scratch, discarding its resumable work (with ``run``, also every
later stage; distill then keeps its teacher shards unless distill itself is
named, since they depend only on the BF16 model and data, which their own
input hash covers). Interrupted stages resume: quantize and refine per layer,
traces per batch, distill per teacher shard and checkpoint. A stage refuses to
run on a missing, disabled or stale upstream output; a later stage whose
inputs changed (another config, or a rerun with a different result) reruns.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import STAGES, ConfigError, load_config, stage_enabled
from .rundir import RunDir, StageNotReady

# Resumable units that are valid only for the inputs they were made from. (The traces stage keys its
# cached batches by content in traces/cache/, which survives config edits; parts/ is the old layout.)
_RESUMABLE = {"traces": ("parts",), "quantize": ("layers",), "refine": ("layers", "streams")}
# Everything --force discards before rerunning a stage.
_FORCE_RESET = {**_RESUMABLE, "traces": ("parts", "cache"),
                "distill": ("checkpoint.pt", "train.jsonl", "TRAIN_INPUTS.json")}
_TEACHER = ("teacher", "TEACHER_INPUTS.json")


def _stage_module(stage: str):
    from .stages import calibrate, distill, endpoints, evaluate, export, quantize, refine, traces
    return {"traces": traces, "calibrate": calibrate, "quantize": quantize, "endpoints": endpoints,
            "refine": refine, "distill": distill, "export": export, "eval": evaluate}[stage]


def _overrides(args) -> list[tuple[tuple[str, ...], object]]:
    """Stage options given as flags (persisted in run.yaml like --set, so they enter the stage hash)."""
    out = []
    if getattr(args, "stage", None) == "export":
        if args.out is not None:
            out.append((("export", "out"), str(Path(args.out).resolve())))   # run.yaml outlives the cwd
        if args.name is not None:
            out.append((("export", "name"), args.name))
        if args.verify is not None:
            out.append((("export", "verify"), bool(args.verify)))
    return out


def resolve_config(args, run: RunDir) -> dict:
    saved = run.load_config()
    if args.config is not None:
        cfg = load_config(args.config, args.set or (), model=args.model)
    elif saved is not None:
        from .config import apply_set, check_keys, deep_merge, DEFAULTS
        cfg = check_keys(deep_merge(DEFAULTS, saved))
        for assignment in args.set or ():
            apply_set(cfg, assignment)
        if args.model is not None:
            cfg["model"]["path"] = args.model
    else:
        cfg = load_config(None, args.set or (), model=args.model)
    for path, value in _overrides(args):
        node = cfg
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
    if not cfg["model"].get("path"):
        raise ConfigError("no model: pass --model or set model.path in the config")
    run.root.mkdir(parents=True, exist_ok=True)
    run.save_config(cfg)
    return cfg


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _prepare(ctx, stage: str, config_hash: str, force) -> None:
    """Drop resumable units made under different inputs (all of them on --force); record the inputs.

    ``force="downstream"``: forced only because an earlier stage was; distill keeps its teacher."""
    from .formats.atomic import atomic_write_json
    d = ctx.run.stage_dir(stage)
    if force:
        teacher = _TEACHER if stage == "distill" and force != "downstream" else ()
        for name in _FORCE_RESET.get(stage, ()) + teacher:
            _remove(d / name)
    if stage not in _RESUMABLE:
        return
    marker = d / "INPUTS.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("config_hash") != config_hash:
        for sub in _RESUMABLE[stage]:
            _remove(d / sub)
    atomic_write_json(marker, {"config_hash": config_hash})


def _output_of(ctx, stage: str) -> Path | None:
    """The file a completed stage produced (``OUTPUTS``, or the path its result names, e.g. the GGUF)."""
    output = ctx.run.output_path(stage)
    if output is None:
        result = (ctx.run.done_info(stage) or {}).get("result")
        if isinstance(result, dict) and result.get("output"):
            output = Path(result["output"])
    return output


def _output_problem(ctx, stage: str) -> str | None:
    """Why a completed stage's recorded output can no longer be used (None if it is there, unchanged)."""
    output = _output_of(ctx, stage)
    if output is None:
        return None
    if not output.is_file():
        return f"its output {output} is missing"
    want = (ctx.run.done_info(stage) or {}).get("output_sha256")
    if want and ctx.digests().file(output) != want:
        return f"its output {output} changed after it ran"
    return None


def execute(ctx, stage: str, *, force=False, **kwargs) -> str:
    """Run one stage unless it is complete with the same config hash and its output is still there,
    unchanged. Returns a status word.

    ``force``: False, True (rerun from scratch) or "downstream" (see ``_prepare``)."""
    from .stages.common import free
    log = ctx.log(stage)
    if not stage_enabled(ctx.cfg, stage):
        ctx.run.clear_done(stage)   # a disabled stage's old output must not feed later stages
        log("disabled in config; skipped")
        return "disabled"
    ctx.require_inputs(stage)
    config_hash = ctx.stage_hash(stage)
    if stage != "eval" and not force and ctx.run.is_done(stage, config_hash):
        problem = _output_problem(ctx, stage)
        if problem is None:
            log(f"up to date ({config_hash[:12]}); skipped")
            return "skipped"
        log(f"complete with this config, but {problem}; rerunning")
    _prepare(ctx, stage, config_hash, force)
    ctx.run.clear_done(stage)
    try:
        result = _stage_module(stage).run(ctx, **kwargs)
    except Exception:          # main() prints one line; keep the traceback of a long stage in its log
        import traceback
        ctx.run.logger(stage, echo=False)("failed:\n" + traceback.format_exc().rstrip())
        raise
    finally:
        free(ctx.device)
    if stage != "eval":
        output = ctx.run.output_path(stage)
        if output is None and isinstance(result, dict) and result.get("output"):
            output = Path(result["output"])
        ctx.run.mark_done(stage, config_hash, result=result,
                          output_sha256=ctx.digests().file(output) if output is not None and output.is_file()
                          else None)
    return "ran"


# docs/adding-a-model.md and the README, "Base model without a chat template"
NO_TEMPLATE_SETS = ("--set traces.enabled=false "
                    "--set 'calibrate.sources=[{kind: wikitext2, n: 128}, {kind: ultrachat, n: 128, min_row: 6000}]' "
                    "--set 'distill.teacher.sources=[{kind: wikitext2, n: 512}, {kind: ultrachat, n: 1536, "
                    "min_row: 6000}]'")


def check_calibration_batch(cfg: dict) -> None:
    """The quantize stage runs the calibration blocks in batches of quantize.batch and refuses a remainder."""
    counts = [max(int(s.get("n") or 0), 0) for s in cfg["calibrate"].get("sources") or ()]
    total, batch = sum(counts), int(cfg["quantize"].get("batch") or 0)
    if batch <= 0 or not total or total % batch == 0:
        return
    last = max(i for i, n in enumerate(counts) if n)
    raise ConfigError(f"calibrate.sources ask for {total} blocks in all, not a multiple of quantize.batch "
                      f"({batch}), which the quantize stage refuses; make the total a multiple of it, e.g. "
                      f"--set calibrate.sources.{last}.n={counts[last] + batch - total % batch}, "
                      f"or set quantize.batch to a divisor of {total}")


def check_chat_template(tok) -> None:
    """The traces stage renders its prompts with the model's chat template."""
    if not getattr(tok, "chat_template", None):
        raise ConfigError("the model's tokenizer has no chat template, so the traces stage cannot render its "
                          "prompts; disable traces and use text sources instead (docs/adding-a-model.md, base "
                          f"model without a chat template): {NO_TEMPLATE_SETS}")


def preflight(ctx, stages=STAGES) -> None:
    """Fail before any work on settings a later stage of ``stages`` (those this command runs) would refuse."""
    from .config import distill_uses_endpoints, upstream
    from .stages.endpoints import ternary_endpoints
    from .stages.export import EXPORT_NEEDS_DISTILLED_ENDPOINTS, check_latent_out, resolve_format
    for stage in stages:              # a stage reading a disabled stage's output (StageContext.require_inputs)
        if stage_enabled(ctx.cfg, stage):
            for u in upstream(ctx.cfg, stage):
                if not stage_enabled(ctx.cfg, u):
                    raise ConfigError(f"{stage} reads the {u} stage's output, but {u} is disabled; "
                                      f"enable it or give {stage} its own path: sources")
    if "quantize" in stages:
        check_calibration_batch(ctx.cfg)
    adapter = ctx.adapter()
    endpoints_on = stage_enabled(ctx.cfg, "endpoints")
    if endpoints_on and "endpoints" in stages:
        ternary_endpoints(ctx.cfg, adapter)    # unknown modes; ternary for a tied table
    if stage_enabled(ctx.cfg, "refine") and "refine" in stages:
        from .stages.refine import check_config
        check_config(ctx.cfg, adapter.n_layers)   # values, layer range, blocks vs the calibration set
    tok = None                                    # loaded once, by the first check that renders a chat template
    if stage_enabled(ctx.cfg, "traces") and "traces" in stages:
        from .stages.traces import checked_chat_kwargs, trace_sources
        tok = ctx.load_tokenizer()
        check_chat_template(tok)
        for src in trace_sources(ctx.cfg):        # also every source's kind-specific settings
            checked_chat_kwargs(adapter, tok, src["thinking"], "traces.thinking" if src["legacy"]
                                else f"{src['where']}.thinking")
    check_trace_supply(ctx, stages)
    if stage_enabled(ctx.cfg, "eval") and "eval" in stages:
        from .eval.reasoning import SUITES
        from .stages.evaluate import check_thinking_modes
        if set(ctx.cfg["eval"].get("suites") or ()) & set(SUITES):
            # a mode the chat template cannot do fails now, not at eval
            check_thinking_modes(ctx.cfg, adapter, tok if tok is not None else ctx.load_tokenizer())
    if stage_enabled(ctx.cfg, "export") and "export" in stages:
        explicit_latent = (ctx.cfg["export"].get("format") or "auto") == "latent"
        # refuses an explicit ptq1_0-gguf the model (its tokenizer, its weights' file format) cannot take
        fmt, _ = resolve_format(ctx.cfg, adapter, None if explicit_latent else ctx.model_dir(weights=False))
        out = ctx.cfg["export"].get("out")
        if out and fmt == "latent":
            check_latent_out(out)
        elif out and Path(out).is_dir():
            raise ConfigError(f"export.out {out} is a folder; the PTQ1_0 export writes a .gguf file")
        if fmt == "ptq1_0-gguf" and not adapter.gguf_endpoints_optional:
            head, emb = ternary_endpoints(ctx.cfg, adapter) if endpoints_on else (False, False)
            if not (head and emb):
                raise ConfigError("the PTQ1_0 GGUF stores the LM head and the token embedding as ternary: enable "
                                  "the endpoints stage with head and embedding ternary, or set export.enabled=false")
            if upstream(ctx.cfg, "export")[0] == "distill" and not distill_uses_endpoints(ctx.cfg):
                raise ConfigError(EXPORT_NEEDS_DISTILLED_ENDPOINTS)


def check_trace_supply(ctx, stages) -> None:
    """The traces, when current, hold the blocks every stage of ``stages`` reads from them (they are
    never capped; stages/traces.py)."""
    from .stages.traces import check_trace_supply as check, trace_demands, trace_tokens
    wanted = [s for s in ("calibrate", "distill") if s in stages]
    if not wanted or not stage_enabled(ctx.cfg, "traces") or not trace_demands(ctx.cfg, wanted):
        return
    if ctx.stale_reason("traces") is not None:
        return                     # checked again once the traces stage has run
    tokens = trace_tokens(ctx.run.stage_dir("traces", create=False))
    if tokens is not None:
        check(ctx.cfg, tokens, ctx.adapter().bos_id(ctx.load_tokenizer()), stages=wanted)


def cmd_run(args) -> int:
    from .stages.common import StageContext
    run = RunDir(args.out)
    ctx = StageContext(resolve_config(args, run), run)
    stop = STAGES.index(args.until) if args.until else len(STAGES) - 1
    preflight(ctx, [s for s in STAGES[:stop + 1] if s not in (args.skip or ())])
    forced = set(args.force or ())
    statuses = {}
    planned = [s for s in STAGES[:stop + 1] if s not in (args.skip or ())]
    for i, stage in enumerate(STAGES[:stop + 1]):
        if stage in (args.skip or ()):
            statuses[stage] = "skipped by --skip"
            continue
        force = True if stage in forced else ("downstream" if any(STAGES.index(f) < i for f in forced) else False)
        statuses[stage] = execute(ctx, stage, force=force)
        if stage == "traces":
            check_trace_supply(ctx, planned[planned.index(stage) + 1:])   # before hours of later stages
    print(json.dumps({"run": str(run.root), "stages": statuses}, indent=1))
    return 0


def cmd_stage(args) -> int:
    from .stages.common import StageContext
    from .stages.evaluate import is_gguf
    if args.run is None:
        if args.stage == "eval" and is_gguf(args.source):
            args.run = str(args.source) + ".eval"
        else:
            raise ConfigError("--run is required (it may be omitted only for eval --from model.gguf)")
    run = RunDir(args.run)
    ctx = StageContext(resolve_config(args, run), run)
    kwargs = {}
    if args.stage == "eval":
        kwargs = {"source": args.source, "suites": args.suite.split(",") if args.suite else None,
                  "install_mode": args.install, "gguf_small_fp32": args.gguf_small_fp32}
    status = execute(ctx, args.stage, force=args.force, **kwargs)
    print(json.dumps({"run": str(run.root), "stage": args.stage, "status": status}))
    return 0


def cmd_inspect(args) -> int:
    from .formats.state import load_state
    path = Path(args.path)
    if path.suffix.lower() == ".gguf":
        from .formats.gguf import GGUFReader
        with GGUFReader(path) as r:
            types = {}
            for t in r.tensors:
                types[t.gtype] = types.get(t.gtype, 0) + 1
            kv = {k: (v if not isinstance(v, (list, str)) or len(v) <= 64 else f"<{type(v).__name__} of {len(v)}>")
                  for k, _, v in r.kv}
            print(json.dumps({"path": str(path), "bytes": r.size, "tensors": len(r.tensors),
                              "tensor_types": {str(k): v for k, v in sorted(types.items())}, "kv": kv}, indent=1))
        return 0
    if path.is_dir() and (path / "openbonsai.json").is_file():   # a latent checkpoint folder
        manifest = json.loads((path / "openbonsai.json").read_text(encoding="utf-8"))
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("verify",)}, indent=1))
        return 0
    if path.is_dir():
        run = RunDir(path)
        info = {"run": str(path), "stages": {s: run.done_info(s) for s in STAGES if run.done_info(s)}}
        print(json.dumps(info, indent=1, default=str))
        return 0
    state = load_state(path)
    cols = {}
    for t in state.tensors.values():
        cols[t.shape[1]] = cols.get(t.shape[1], 0) + 1
    weights = sum(t.codes.numel() for t in state.tensors.values())
    print(json.dumps({"path": str(path), "tensors": len(state.tensors), "weights": weights,
                      "block": state.block, "group": state.group, "sign_widths": sorted(state.signs),
                      "tensors_by_input_width": cols,
                      "ptq1_0_bytes": weights // 128 * 28, "meta": state.meta}, indent=1))
    return 0


def _debug(p: argparse.ArgumentParser, default) -> None:
    p.add_argument("--debug", action="store_true", default=default,
                   help="show the traceback of an error (also OPENBONSAI_DEBUG=1)")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="HF model id or local directory (sets model.path)")
    p.add_argument("--config", help="YAML config (defaults to the run's saved run.yaml)")
    p.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override one config value")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="openbonsai", description="Ternary (PTQ1_0) compression of dense LLMs.")
    ap.add_argument("--version", action="version", version=f"openbonsai {__version__}")
    _debug(ap, False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every stage into one run directory")
    r.add_argument("--out", required=True, help="run directory")
    _common(r)
    r.add_argument("--until", choices=STAGES, help="stop after this stage")
    r.add_argument("--force", action="append", choices=STAGES, help="rerun this stage and every later one")
    r.add_argument("--skip", action="append", choices=STAGES, help="do not run this stage")
    _debug(r, argparse.SUPPRESS)       # also after the command; SUPPRESS keeps a --debug given before it
    r.set_defaults(func=cmd_run)
    for stage in STAGES:
        s = sub.add_parser(stage, help=f"run the {stage} stage")
        s.add_argument("--run", required=stage != "eval", default=None, help="run directory")
        _common(s)
        s.add_argument("--force", action="store_true", help="rerun even if up to date")
        if stage == "eval":
            s.add_argument("--from", dest="source", default=None,
                           help="bf16 | quantize | endpoints | refine | distill | state.safetensors | legacy.pt[,more.pt] "
                                "| model.gguf")
            s.add_argument("--suite", default=None, help="comma-separated: dev, extended, compare, teacher, gsm8k, humaneval "
                                "(the last two: eval.reasoning.*)")
            s.add_argument("--install", choices=("auto", "dense", "ternary"), default="auto",
                           help="dense weights or ternary modules (auto: ternary for distill and GGUF)")
            s.add_argument("--gguf-small-fp32", action="store_true",
                           help="GGUF source: keep 1-D F32 tensors (norms, A_log, dt_bias) in FP32 as Prism does")
        if stage == "export":
            s.add_argument("--out", default=None,
                           help="output .gguf, or the latent checkpoint folder itself: new, empty or a previous "
                                "latent checkpoint (replaced); any other folder is refused, never deleted "
                                "(default: <run>/export/<name>.gguf or <run>/export/<name>/)")
            s.add_argument("--name", default=None, help="general.name (default: OpenBonsai-<model>-PTQ1_0)")
            s.add_argument("--verify", dest="verify", action="store_true", default=None,
                           help="verify the written file (the default)")
            s.add_argument("--no-verify", dest="verify", action="store_false", help="skip verification")
        _debug(s, argparse.SUPPRESS)
        s.set_defaults(func=cmd_stage, stage=stage)
    i = sub.add_parser("inspect", help="summarize a state file, GGUF, latent checkpoint folder or run directory")
    i.add_argument("path")
    _debug(i, argparse.SUPPRESS)
    i.set_defaults(func=cmd_inspect)
    return ap


def _one_line(error: BaseException) -> str:
    text = " ".join(str(error).split()) or type(error).__name__
    return text if isinstance(error, (ConfigError, StageNotReady)) else f"{text} ({type(error).__name__})"


def main(argv=None) -> int:
    """Errors a user can act on (config, plan, data, missing inputs) print one line and a hint and return 2;
    with --debug or OPENBONSAI_DEBUG=1 they propagate with their traceback."""
    args = build_parser().parse_args(argv)
    from .stages.export import ExportError
    debug = bool(getattr(args, "debug", False)) or os.environ.get("OPENBONSAI_DEBUG", "") not in ("", "0")
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, NotImplementedError, StageNotReady, ExportError) as error:
        # ValueError covers ConfigError, ContaminationError, GGUFError, UnsupportedGeometry, SuiteNotApplicable
        if debug:
            raise
        print(f"openbonsai: error: {_one_line(error)}", file=sys.stderr)
        print("openbonsai: hint: rerun with --debug (or OPENBONSAI_DEBUG=1) for the traceback; if a stage "
              "failed, its log <run>/logs/<stage>.log has it too", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
