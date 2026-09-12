# Real bitstreams on CPU

DCVC-UF cannot write one without a GPU. Its `compress()` goes through
`inference_extensions_cuda` — CUTLASS fused kernels — and raises
`NotImplementedError` when that import fails. There is no CPU fallback, and the
CUDA half does the network *and* the entropy coding, so there is nothing to
borrow.

**DCVC-RT (CVPR 2025), its direct predecessor, can.** Three differences matter:

| | DCVC-UF | DCVC-RT |
|---|---|---|
| entropy coding | inside the CUDA proxy | plain Python over the CPU rANS coder |
| missing CUDA extension | `NotImplementedError` | *"fallback to pytorch"* — by design |
| what still needs CUDA | everything in `compress()` | only stream/event scheduling |

`compress()` in DCVC-RT is readable Python, and it is the reference for what
UF's CUDA proxy does internally:

```python
self.entropy_coder.reset()
self.bit_estimator_z.encode_z(z_hat_write, qp)
self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)   # the 4 spatial-prior steps
self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
self.gaussian_encoder.encode_y(y_q_w_2, s_w_2)
self.gaussian_encoder.encode_y(y_q_w_3, s_w_3)
self.entropy_coder.flush()
bit_stream = self.entropy_coder.get_encoded_stream()
```

The only CUDA left is a stream and an event used to overlap the decoder network
with the entropy coding — a latency trick, not computation. On CPU the two halves
just run in sequence. [`cpu_cuda_shim.py`](cpu_cuda_shim.py) no-ops exactly those
four entry points (`torch.cuda.Event`, `torch.cuda.stream`,
`torch.cuda.synchronize`, `CompressionModel.get_cuda_stream`) and nothing else,
and does nothing at all when CUDA is present — the same script keeps upstream's
overlap on a GPU box.

## Run

```bash
make build          # builds the new `bitstream` image alongside the others
make bitstream
```

Verified on kodim19 (512×768) on an arm64 Mac, CPU only: a real `.bin` per qp in
`outputs/experiments/08-real-bitstream/`, and the decoder reconstructs from the
bytes alone with `max abs diff = 0` at every rate point.

Without `weights/dcvc-rt/cvpr2025_image.pth.tar` it runs with random weights: the
bitstream is still real and the round-trip still exact, but the byte counts mean
nothing. `make weights` prints where to get the checkpoints.

## Why its own image

Both repos build a pybind11 module named `MLCodec_extensions_cpp`, and they are
not compatible: UF's `pmf_to_quantized_cdf` takes one argument, RT's takes two
(`pmf, precision`). Same name, different ABI — installing one shadows the other,
and the mismatch surfaces as a `TypeError` deep inside `update()`. One image
each is the only clean separation. The `bitstream` service's Dockerfile asserts
it got the right one at build time.

## If you want this for DCVC-UF specifically

Two options, in order of effort:

1. **A CUDA machine.** Build both extensions per the upstream README
   (`third_party/DCVC/README.md`) and use `test_video.py --write_stream 1`.
   This is the supported path and the only one that produces bitstreams
   interoperable with upstream's decoder.
2. **Port the orchestration.** Everything needed is already exposed on CPU —
   `MLCodec_extensions_cpp` binds a complete `RansEncoder`/`RansDecoder`
   (`encode_y`, `encode_z`, `flush`, `get_encoded_stream`, `set_cdf`) — and
   DCVC-RT's `compress()`/`decompress()` above show the exact call sequence. What
   has to be replicated is UF's symbol ordering, its scale→CDF index mapping and
   its `skip_thres` handling, and the result would only interoperate with
   upstream's CUDA decoder if it matches byte for byte. Doable, not small.
