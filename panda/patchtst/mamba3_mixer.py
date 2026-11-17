"""PAT-318 control: official Mamba-3 (state-spaces/mamba @ e9594ce, ../mamba_official, SISO triton kernels) as Panda's
temporal token mixer. The mamba_ssm package __init__ imports the compiled CUDA selective-scan extension, which Mamba-3
does not use; the package is therefore registered without running its __init__, and only the Mamba-3 module is loaded.

Bidirectional with tied weights (as in Caduceus' BiMamba): one Mamba-3 runs on x and on x reversed, outputs summed after
re-reversal. Panda's temporal attention is non-causal; path / fox walk both directions as well."""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[3] / "mamba_official" / "mamba_ssm"
if "mamba_ssm" not in sys.modules:
    _pkg = types.ModuleType("mamba_ssm")
    _pkg.__path__ = [str(_ROOT)]
    sys.modules["mamba_ssm"] = _pkg
from mamba_ssm.modules.mamba3 import Mamba3  # noqa: E402


class BiMamba3(nn.Module):
    def __init__(self, d_model: int, d_state: int = 128, expand: int = 1, headdim: int = 64, chunk_size: int = 64):
        super().__init__()
        self.mixer = Mamba3(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, chunk_size=chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        y = self.mixer(torch.cat([x, x.flip(1)], 0))
        return y[:b] + y[b:].flip(1)
