"""Let DCVC-RT's compress()/decompress() run on a machine with no CUDA.

DCVC-RT's bitstream path is already pure PyTorch plus the CPU rANS coder in
MLCodec_extensions_cpp. The only thing standing between it and a CPU run is
*scheduling*: compress() creates a CUDA event and a second CUDA stream so the
decoder network overlaps the entropy coding, then synchronises at the end.

    cuda_event = torch.cuda.Event(); cuda_event.record()
    x_hat = self.dec(...)
    with torch.cuda.stream(self.get_cuda_stream(...)):
        cuda_event.wait()
        ... encode_z / encode_y / flush ...
    torch.cuda.synchronize(device)

None of that is computation. On CPU the two halves simply run in sequence, which
is what a null stream and a no-op event give you. This shim replaces exactly
those four entry points and nothing else -- the network, the quantisation and
the arithmetic coding are untouched, so the bytes produced are the real thing.

    with cpu_cuda_shim():
        enc = net.compress(x, qp)
        dec = net.decompress(enc["bit_stream"], sps, qp)
"""
import contextlib

import torch


class _NoOpEvent:
    """Stands in for torch.cuda.Event: ordering is implicit when work is serial."""

    def record(self, *a, **k):
        pass

    def wait(self, *a, **k):
        pass

    def synchronize(self, *a, **k):
        pass

    def query(self):
        return True

    def elapsed_time(self, other):
        return 0.0


@contextlib.contextmanager
def cpu_cuda_shim(force=False):
    """No-op the CUDA scheduling calls in the codec's bitstream path.

    Does nothing when CUDA is actually present, so the same script runs unchanged
    on a GPU box and keeps upstream's stream overlap there. Pass force=True only
    to test the CPU path on a machine that has a GPU.
    """
    if torch.cuda.is_available() and not force:
        yield
        return

    from src.models.common_model import CompressionModel

    saved = {
        "Event": torch.cuda.Event,
        "stream": torch.cuda.stream,
        "synchronize": torch.cuda.synchronize,
        "get_cuda_stream": CompressionModel.get_cuda_stream,
    }
    torch.cuda.Event = _NoOpEvent
    torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()
    torch.cuda.synchronize = lambda *a, **k: None
    CompressionModel.get_cuda_stream = lambda self, device, idx=0, priority=0: None
    try:
        yield
    finally:
        torch.cuda.Event = saved["Event"]
        torch.cuda.stream = saved["stream"]
        torch.cuda.synchronize = saved["synchronize"]
        CompressionModel.get_cuda_stream = saved["get_cuda_stream"]
