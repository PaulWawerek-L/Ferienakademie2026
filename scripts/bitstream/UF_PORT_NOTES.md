# Porting a CPU bitstream path to DCVC-UF — findings

Status: **done** — image in [`uf_codec.py`](uf_codec.py) (verified by
[`uf_selftest.py`](uf_selftest.py)), video in [`uf_video_codec.py`](uf_video_codec.py)
for LD, HTS and HTL (verified by [`uf_video_selftest.py`](uf_video_selftest.py)).
Both self-tests run on every `make smoke`.
Everything below is established from the source.

DCVC-RT (UF's predecessor) is referred to below because its readable Python
`compress()` was the starting reference for the port. It is no longer part of
this repo -- the course uses UF, which now writes real bitstreams itself -- but
its source remains in the upstream checkout at
`third_party/DCVC/DCVC-family/DCVC-RT`.

## The model swap does not work

DCVC-UF and DCVC-RT are different networks, so UF weights cannot be loaded into
RT's code:

| | |
|---|---|
| RT `DMCI` parameters | 407 |
| UF checkpoint keys | 398 |
| shared key names | 257 |
| **of those, matching shapes** | **45 (~11 %)** |

Concretely: RT's decoder trunk is 368 channels against UF's 384 (FFN 1472 vs
1536), and the factorised prior is parameterised differently — UF stacks
`bit_estimator_z.a/b/h` as `(64, 128, 3or4)`, RT splits it into `f1…f4`
sub-modules of `(64, 128, 1, 1)`. The video models differ more fundamentally
still: UF's HT models code a chunk of 8 frames (`g_frame_delay = 8`), RT codes
one at a time.

## What UF already gives us

`net.update(skip_thres)` **already registers both CDF tables with the CPU rANS
coder** — this was the part expected to be hard, and upstream does it:

```python
GaussianEncoder.update: self.entropy_coder.set_cdf(*self.get_cdf_info(), 1)   # group 1 = y
BitEstimator.update:    self.entropy_coder.set_cdf(*self.get_cdf_info(), 0)   # group 0 = z
```

## The wire format, read off the C++

`encode_y` takes one `int16` array; each element packs both the symbol and which
CDF to use (`rans.cpp`, `encode_y_internal`):

```c
cdf_idx = combined_symbol & 0xff;                    // low byte  = scale bin, 0..127
s       = (int8_t)(combined_symbol >> 8);            // high byte = signed quantised value
```

so in Python: `combined = y_q * 256 + scale_idx`, as `int16`. `y_q` must fit in
an `int8`, which is why `process_with_mask` clamps to [-128, 127]. The reordering
to 0, 1, -1, 2, -2 … happens inside `encode_one_symbol`
(`value = abs(s) * 2 - (s > 0)`), so pass the **raw signed value**, not a
reordered index.

Scale bin: `idx = clamp(round((log(scale) - log(scale_min)) * log_step_recip), 0, 127)`
with `scale_min = 0.11`, `scale_max = 16.0`, `scale_level = 128`.

`encode_z` takes `int8` symbols with `cdf_idx = (i % ch) + cdf_offset`, so the z
tensor must be flattened **channel-minor** — permute `(B,C,H,W) → (B,H,W,C)` —
with `cdf_offset = qp * channel`.

## The one blocker, now fixed

UF's `bind.cpp` bound the decoder's `set_stream`/`decode_y`/`decode_z`/`set_cdf`
but **no accessor for the decoded symbols**: `get_decoded_tensor_cpp()` exists
only for C++ callers (the CUDA proxy links it directly) and never reached Python.
Encoding was fully bound; decoding could run but the values could not be read.

`docker/dcvc/patch-expose-decoder.sh` adds that one binding as a lambda wrapping
the existing accessor — a pure addition, no header or implementation change. The
Dockerfile applies it before building and asserts it took. Verified:

```
encoder: encode_y, encode_z, flush, get_encoded_stream, reset, set_cdf, ...
decoder: decode_y, decode_z, get_decoded_tensor, set_cdf, set_stream, ...
```

## Two things the RT code would have got wrong

Reading RT's `compress()` suggested the call order z, y0, y1, y2, y3. **UF's
library needs the reverse.** RT's coder buffers; UF's writes each call
immediately, and rANS is last-in-first-out. UF's proxy worker
(`DMCIProxy::worker` in `dmci_proxy.cpp`) encodes y steps 3, 2, 1, 0 and then z,
so that the decoder can read z, then step 0, then step 1 — the order the
autoregressive prior needs.

The decoder's output buffer is **reused and allocated at twice the requested
size** (`RansDecoder::decode_y`), so the accessor returns more than was decoded.
Every read must be sliced to the count just decoded, or stale symbols from an
earlier step leak into this one.

Two smaller ones: UF's Python `process_with_mask` does not clamp `y_q`, but the
symbol must fit the int8 half of the packed int16, so the codec clamps and derives
the reconstruction from the clamped value; and upstream's scale index rounds
**down** (`to_uint8` → `__half2uint_rd`), from rounded constants
(`LOG_SCALE_MIN = -2.2073`) that the codec uses verbatim.

## Verification

- bit-exact decode, using a separately constructed model;
- one flipped payload byte changes the reconstruction (the decoder really reads the stream);
- at `skip_thres = 0`, identical to upstream `forward_one_frame(..., recon_only=True)`
  at qp 0, 32 and 63 — the check that catches a bug shared by encoder and decoder,
  which a round trip alone cannot.

With `skip_thres = 0.15` the reconstruction differs from `forward()` (max 0.27 at
qp 0, 0.06 at qp 63) — that is the skipped latents, and the difference vanishes
at threshold 0.

## Not interoperable with upstream's decoder

The rANS payload follows upstream's symbol format, but the container header is
this codec's own, and the proxy computes scale indices in float16 where this uses
float32. A stream written here decodes here; do not expect upstream's CUDA
decoder to read it.

## Video

The inter models reuse everything above — packing, scale index, clamping, the
decoder-buffer slicing — and add three things.

**The structures are coded differently** (`dmc_*_proxy.cpp`, `compress` and `worker`):

| structure | spatial prior | scales | `encode_y` calls |
|---|---|---|---|
| LD | `forward_prior_2x`, means only | hyperprior | **one**, whole y, NHWC |
| HTS | `forward_prior_4x`, means only | hyperprior | **one**, whole y, NHWC |
| HTL | `forward_prior_4x`, scales + means | re-predicted per step | four, steps 3, 2, 1, 0 |

For LD and HTS the proxy accumulates every step's `y_q` into one full tensor,
indexes it with the hyperprior's scales, and writes it in a single call. The
decoder can then decode every symbol at once and only runs the spatial prior to
restore means step by step. Copying the image codec's per-step layout would
still round-trip, but would not be what upstream writes.

**Quantisation step.** Video takes it per element from the prior parameters
(`separate_prior_video`: `quant_step = clamp_min(q, 0.5)`), not per qp from a
table. The proxy computes `y / quant_step`; the codec keeps Python's
`y * (1 / quant_step)` so it can be compared against `forward()` bit for bit.

**The decoded-picture buffer.** `ref_feature`, `memory` and `ctx` are advanced by
each frame's decoded feature. Encoder and decoder hold separate model objects and
stay in step only because `y_hat` is reproduced exactly. The proxies update the
buffer eagerly after encoding and lazily before decoding; the values are equal,
and the codec follows Python's lazy `apply_feature_adaptor()` on both sides. The
intra frame seeds it with `pixel_unshuffle(x_hat, 8)` of the **padded**
reconstruction, as `add_ref_feature_from_frame` does.

**Chunks.** HT models code 8 frames per call; a short final chunk repeats its last
frame, as `test_video.py` does, and the decoder drops the extras. Those padding
frames are really coded, which is why HTS's real bytes sit ~1.56 % above an
estimate that bills the last chunk pro rata.

Verification is the same three checks, per structure: every frame bit-exact from
separate objects; a flipped byte in an inter packet leaves the intra frame
untouched and changes the inter frames; and at `skip_thres = 0` equality with
the `forward_one_frame` pipeline. LD, HTS and HTL all pass.
