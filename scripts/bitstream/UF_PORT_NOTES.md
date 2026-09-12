# Porting a CPU bitstream path to DCVC-UF — findings

Status: **groundwork done and verified; the Python codec itself is still to be
written.** Everything below is established fact from the source, not plan.

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

## What remains

Write, against UF's own modules:

1. `build_indexes_encoder/decoder` — the scale→bin mapping above.
2. `GaussianEncoder.encode_y` / `decode_y`, `BitEstimator.encode_z` / `decode_z`.
3. `compress_prior_4x` / `decompress_prior_4x` — UF has only `forward_prior_4x`;
   the encode-side variant must additionally return each of the four
   spatial-prior steps' `y_q` and `s_hat` packed with `single_part_for_writing_4x`
   (`(x0+x1)+(x2+x3)` over channel quarters). UF's masks are `bool` where RT's
   carry a dtype, and UF threads `q_enc`/`q_dec` that RT does not — those two are
   the fiddly parts.
4. `DMCI.compress` / `decompress`.

Start with `skip_thres = 0` (UF's default): nothing is skipped, so encoder and
decoder agree trivially. Add the skip path only once the round trip is exact.

**Correctness check:** decoding must reproduce the encoder's `x_hat` with
`max abs diff == 0`. It is self-verifying — any mismatch in symbol order, bin
index or mask phase shows up immediately.

Interoperability with upstream's CUDA decoder is **not** implied: that needs the
header/SPS layout to match byte for byte as well.
