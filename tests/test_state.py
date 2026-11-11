"""Run-state files: round trips, determinism, atomic writes, legacy research states."""
import pytest
import torch

from openbonsai.core.geometry import derive_geometry, make_signs
from openbonsai.formats.state import (TernaryState, TernaryTensor, legacy_to_state, load_legacy_pt, load_scales,
                                      load_signs, load_state, save_scales, save_signs, save_state, state_to_bytes)


def make_state(seed=0, keys=("layers.0.mlp.up_proj", "layers.0.mlp.down_proj", "lm_head")):
    g = torch.Generator().manual_seed(seed)
    geo = derive_geometry((128, 256), 128)
    signs = make_signs(geo)
    shapes = {"layers.0.mlp.up_proj": (256, 128), "layers.0.mlp.down_proj": (128, 256), "lm_head": (320, 128)}
    tensors = {k: TernaryTensor(torch.randint(-1, 2, shapes[k], generator=g).to(torch.int8),
                                torch.rand(shapes[k][0], shapes[k][1] // 128, generator=g).half())
               for k in keys}
    return TernaryState(tensors, signs, geo.block, geo.group, {"stage": "test", "n": 1}).validate()


@pytest.mark.parametrize("packed", [True, False])
def test_round_trip_is_exact_and_bytes_are_deterministic(tmp_path, packed):
    state = make_state()
    path = save_state(tmp_path / "s.safetensors", state, packed=packed)
    back = load_state(path)
    assert back.block == 128 and back.group == 128 and back.meta == {"stage": "test", "n": 1}
    assert sorted(back.tensors) == sorted(state.tensors)
    for k, t in state.tensors.items():
        assert torch.equal(back.tensors[k].codes, t.codes) and torch.equal(back.tensors[k].scales, t.scales)
    for w, s in state.signs.items():
        assert torch.equal(back.signs[w], s)
    assert path.read_bytes() == state_to_bytes(make_state(), packed=packed)
    assert not list(tmp_path.glob("*.tmp"))


def test_packed_codes_are_two_bits_per_weight(tmp_path):
    state = make_state()
    packed = save_state(tmp_path / "p.safetensors", state).stat().st_size
    unpacked = save_state(tmp_path / "u.safetensors", state, packed=False).stat().st_size
    weights = sum(t.codes.numel() for t in state.tensors.values())
    assert abs((unpacked - packed) - (weights - weights // 4)) < 64   # header length differs slightly


def test_merge_scales_and_validation():
    a, b = make_state(keys=("layers.0.mlp.up_proj",)), make_state(keys=("lm_head",))
    merged = a.merged(b)
    assert sorted(merged.tensors) == ["layers.0.mlp.up_proj", "lm_head"]
    with pytest.raises(ValueError, match="overlap"):
        merged.merged(a)
    other = make_state(keys=("lm_head",))
    other.signs = {w: -s for w, s in other.signs.items()}
    with pytest.raises(ValueError, match="different signs"):
        a.merged(other)
    new = merged.with_scales({"lm_head": torch.full((320, 1), 0.25)})
    assert new.tensors["lm_head"].scales.dtype == torch.float16 and (new.tensors["lm_head"].scales == 0.25).all()
    with pytest.raises(ValueError, match="unknown"):
        merged.with_scales({"nope": torch.ones(1, 1)})
    with pytest.raises(ValueError, match="do not match"):
        merged.with_scales({"lm_head": torch.ones(320, 2)})
    with pytest.raises(ValueError):
        TernaryTensor(torch.zeros(2, 128, dtype=torch.int8), torch.zeros(2, 3).half())
    keys = [k for k, *_ in merged.records()]
    assert keys == sorted(keys)


def test_signs_and_scales_files(tmp_path):
    state = make_state()
    save_signs(tmp_path / "signs.safetensors", state.signs, block=128)
    signs, block = load_signs(tmp_path / "signs.safetensors")
    assert block == 128 and all(torch.equal(signs[w], state.signs[w]) for w in state.signs)
    scales = {k: t.scales for k, t in state.tensors.items()}
    save_scales(tmp_path / "scales.safetensors", scales, {"steps": 3})
    back = load_scales(tmp_path / "scales.safetensors")
    assert all(torch.equal(back[k], v) for k, v in scales.items())
    with pytest.raises(ValueError, match="not an"):
        load_state(tmp_path / "scales.safetensors")


def _legacy(path, shared):
    """A research/experiments ternary_state.pt in its exact on-disk layout."""
    g = torch.Generator().manual_seed(1)
    modules = {}
    for li, name, (o, i) in ((0, "mlp.up_proj", (256, 128)), (0, "mlp.gate_proj", (256, 128)), (1, "mlp.down_proj", (128, 256))):
        seed = i if shared else 5090 + 1000 * li + sum(map(ord, name))
        gs = torch.Generator().manual_seed(seed)
        modules[f"layers.{li}.{name}"] = {
            "codes": torch.randint(-1, 2, (o, i), generator=g).to(torch.int8),
            "scales": torch.rand(o, i // 128, generator=g).half(), "seed": seed,
            "signs": ((torch.randint(0, 2, (i,), generator=gs) * 2 - 1).float()).to(torch.int8),
            "block": 128, "rel_mse": 0.3, "h_rel_err": 0.05}
    torch.save({"args": {"block": 4096, "shared_signs": shared}, "modules": modules}, path)
    return modules


def test_legacy_states_load_and_only_shared_signs_convert(tmp_path):
    per_module = _legacy(tmp_path / "per_module.pt", shared=False)
    legacy = load_legacy_pt(tmp_path / "per_module.pt")
    assert sorted(legacy) == sorted(per_module)
    assert torch.equal(legacy["layers.0.mlp.up_proj"].signs, per_module["layers.0.mlp.up_proj"]["signs"])
    with pytest.raises(ValueError, match="per-module signs"):
        legacy_to_state(legacy)
    _legacy(tmp_path / "shared.pt", shared=True)
    state = legacy_to_state(load_legacy_pt(tmp_path / "shared.pt"))
    assert sorted(state.signs) == [128, 256] and state.block == 128
    assert torch.equal(state.signs[128], make_signs(derive_geometry((128, 256), 128))[128])


def test_legacy_states_without_stored_signs_use_the_seed(tmp_path):
    """The 256-sample per-module run predates stored signs: modules carry only ``seed``."""
    modules = _legacy(tmp_path / "with.pt", shared=False)
    for st in modules.values():
        del st["signs"]
    torch.save({"args": {}, "modules": modules}, tmp_path / "seed_only.pt")
    with_signs, seed_only = load_legacy_pt(tmp_path / "with.pt"), load_legacy_pt(tmp_path / "seed_only.pt")
    for k in with_signs:
        assert torch.equal(seed_only[k].signs, with_signs[k].signs) and seed_only[k].signs.dtype == torch.int8
    del modules["layers.0.mlp.up_proj"]["seed"]
    torch.save({"modules": modules}, tmp_path / "neither.pt")
    with pytest.raises(ValueError, match="neither signs nor seed"):
        load_legacy_pt(tmp_path / "neither.pt")


def test_legacy_signs_that_disagree_with_the_seed_are_refused(tmp_path):
    """The research scripts rebuilt the rotation from ``seed``; stored signs must be that vector."""
    modules = _legacy(tmp_path / "ok.pt", shared=False)
    load_legacy_pt(tmp_path / "ok.pt")
    st = modules["layers.0.mlp.up_proj"]
    st["signs"] = -st["signs"]
    torch.save({"args": {}, "modules": modules}, tmp_path / "flipped.pt")
    with pytest.raises(ValueError, match="differ from signs_from_seed"):
        load_legacy_pt(tmp_path / "flipped.pt")
