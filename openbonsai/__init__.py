"""OpenBonsai: ternary compression of dense language models into Prism's PTQ1_0 layout.

Method: signed block-Hadamard rotation, sequential GPTQ ternary rounding on
calibration data that includes the model's own traces, ternary embedding and
head, then distillation of the group scales against the original model.
Entry point: ``openbonsai`` (openbonsai/cli.py).
"""
__version__ = "0.2.0"
