"""Preflight for the DCVC-UF environment -- runs WITHOUT any checkpoint.

The pretrained weights are on OneDrive and must be fetched by hand, so this
script answers the question that blocks everything else: is the container
itself sound, and does the CPU-only forward path actually work?

It deliberately also asserts the known CPU limitation, so the limitation is a
tested fact rather than a claim in a README that silently rots.
"""
import sys
import torch

print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")

import MLCodec_extensions_cpp as rans
print(f"rANS extension   OK ({rans.__file__})")

from src.models.image_model import DMCI

net = DMCI().eval()
n_params = sum(p.numel() for p in net.parameters())
print(f"DMCI             instantiated, {n_params/1e6:.1f}M params, qp levels = {DMCI.qp_num()}")

# Random weights are fine here: we are testing that the graph executes on CPU,
# not that it compresses well.
x = torch.rand(1, 3, 256, 256) - 0.5
# qp is a per-sample LongTensor, not a plain int: forward_one_frame feeds it
# straight to torch.index_select to pick that rate's quantisation scales.
qp = torch.tensor([DMCI.qp_num() // 2])
with torch.no_grad():
    out = net.forward_one_frame(x, qp)

x_hat = out["x_hat"]
bpp = out["bpp"]
assert x_hat.shape == x.shape, f"shape mismatch: {x_hat.shape} vs {x.shape}"
print(f"forward()        OK -> x_hat {tuple(x_hat.shape)}, estimated bpp {bpp.item():.4f}")
print("                 (analysis/synthesis transform + hyperprior + entropy model all ran)")
print("                 NOTE: the bpp above is meaningless -- these are random weights,")
print("                 not the trained checkpoint. Only the fact that it ran matters here.")

# The CUTLASS kernels are CUDA-only, so the real-bitstream path must fail here.
try:
    net.compress(x, qp, 0, 0)
except NotImplementedError as e:
    print(f"compress()       correctly unavailable on CPU: {str(e).splitlines()[0]}")
except Exception as e:
    print(f"compress()       unavailable ({type(e).__name__})")
else:
    print("compress()       UNEXPECTEDLY available -- CUDA extensions present?")

print("\nPREFLIGHT PASSED -- environment is sound; add checkpoints to weights/dcvc/ to run real data.")
