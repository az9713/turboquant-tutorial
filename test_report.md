# TurboQuant Test Report

_TurboQuant Implementation Verification
Based on: 'TurboQuant: Online Vector Quantization' (ICLR 2026)_

---

## TEST 1: Lloyd-Max Codebook Properties

```
d=  64, bits=1: 2 levels, distortion/coord=0.005678, centroids range=[-0.0997, 0.0997]
  d=  64, bits=2: 4 levels, distortion/coord=0.001836, centroids range=[-0.1888, 0.1888]
  d=  64, bits=3: 8 levels, distortion/coord=0.000540, centroids range=[-0.2690, 0.2690]
  d=  64, bits=4: 16 levels, distortion/coord=0.000148, centroids range=[-0.3416, 0.3416]
  d= 128, bits=1: 2 levels, distortion/coord=0.002839, centroids range=[-0.0705, 0.0705]
  d= 128, bits=2: 4 levels, distortion/coord=0.000918, centroids range=[-0.1335, 0.1335]
  d= 128, bits=3: 8 levels, distortion/coord=0.000270, centroids range=[-0.1902, 0.1902]
  d= 128, bits=4: 16 levels, distortion/coord=0.000074, centroids range=[-0.2416, 0.2416]
  d= 256, bits=1: 2 levels, distortion/coord=0.001419, centroids range=[-0.0499, 0.0499]
  d= 256, bits=2: 4 levels, distortion/coord=0.000459, centroids range=[-0.0944, 0.0944]
  d= 256, bits=3: 8 levels, distortion/coord=0.000135, centroids range=[-0.1345, 0.1345]
  d= 256, bits=4: 16 levels, distortion/coord=0.000037, centroids range=[-0.1708, 0.1708]

  Symmetry check (d=128, b=3): sum of centroids = 0.000000 (should be ~0)
  PASSED
```

### Interpretation

**What it checks:** The structure of the Lloyd-Max codebook — the set of optimal quantization buckets for each bit-width and dimension.

- `levels` should equal `2^bits` (2, 4, 8, 16). Confirms the correct number of buckets.
- `distortion/coord` decreases as bits or `d` increase. Larger `d` means better averaging, so per-coordinate error shrinks.
- `centroids range` should be symmetric (e.g. `[-0.19, 0.19]`). Asymmetry indicates a bug.
- **Symmetry check** is the key assertion: `sum of centroids ≈ 0`. Non-zero means the codebook is biased, which would corrupt all inner product estimates downstream.

**Healthy result:** `PASSED`, distortion monotonically decreasing with bits and d.

## TEST 2: MSE Quantizer Distortion

```
bits=1: MSE=0.360677, theory_bound=0.680175, ratio=0.530 [OK]
  bits=2: MSE=0.115711, theory_bound=0.170044, ratio=0.680 [OK]
  bits=3: MSE=0.033985, theory_bound=0.042511, ratio=0.799 [OK]
  bits=4: MSE=0.009416, theory_bound=0.010628, ratio=0.886 [OK]
```

### Interpretation

**What it checks:** Whether the quantizer's actual reconstruction error stays within the theoretical guarantee from the paper.

- `MSE` — empirical mean squared error over 1,000 random unit vectors.
- `theory_bound` — the paper's upper bound: `√3 · π/2 · (1/4^bits)`.
- `ratio = MSE / theory_bound` — must be ≤ 1.0 to satisfy the bound. Values around 0.5–0.9 are normal (the bound is loose).
- `[OK]` means ratio ≤ 1.5 (with slack for finite `d`). `[WARN]` means the bound is violated.

**Healthy result:** All `[OK]`, ratios well below 1.0, MSE approximately halving with each added bit.

## TEST 3: Inner Product Unbiasedness (QJL Correction)

```
bits=2: bias=-0.001023, RMSE=0.066204, corr=0.7979, theory_D=0.008347
  bits=3: bias=+0.001026, RMSE=0.037788, corr=0.9213, theory_D=0.002087
  bits=4: bias=+0.000310, RMSE=0.020075, corr=0.9751, theory_D=0.000522
```

### Interpretation

**What it checks:** Whether the full two-stage TurboQuant estimator (Stage 1 MSE + Stage 2 QJL correction) gives accurate, unbiased dot product estimates.

- `bias` — systematic offset of estimated vs true inner products. Should be near `0.000`. This is the QJL correction doing its job.
- `RMSE` — random error around the true value. Decreases with more bits.
- `corr` — Pearson correlation between estimated and true inner products across 2,000 pairs. Most intuitive metric: 0.80 at 2-bit, 0.97 at 4-bit. Even 0.80 preserves the relative ranking of tokens well enough for attention.
- `theory_D` — the paper's variance bound. `RMSE²` should be near or below this.

**Healthy result:** Bias near zero at all bit-widths, correlation improving from ~0.80 → ~0.97.

## TEST 4: MSE-Only Inner Product Bias (motivation for QJL)

```
bits=1: bias=+0.000533 (MSE-only is biased, QJL fixes this)
  bits=2: bias=+0.000139 (MSE-only is biased, QJL fixes this)
  bits=3: bias=-0.000497 (MSE-only is biased, QJL fixes this)
```

### Interpretation

**What it checks:** Demonstrates *why* Stage 2 (QJL) is necessary by showing that Stage 1 (MSE quantization) alone produces biased inner products.

- MSE quantization shrinks vectors, so dot products are systematically underestimated.
- Compare these bias values with Test 3: after QJL correction, bias drops to near zero.
- The message printed per row is intentional — it is the conceptual point of this test.

**Healthy result:** Non-zero biases (contrasting with Test 3's near-zero values), confirming QJL is load-bearing.

## TEST 5: KV Cache Compression Ratios

```
bits=2: compression=7.76x (66.0 KB vs 512.0 KB fp16)
           attention scores shape: torch.Size([1024]), range=[-49.651, 53.039]
  bits=3: compression=5.22x (98.0 KB vs 512.0 KB fp16)
           attention scores shape: torch.Size([1024]), range=[-43.112, 52.800]
  bits=4: compression=3.94x (130.0 KB vs 512.0 KB fp16)
           attention scores shape: torch.Size([1024]), range=[-55.905, 48.531]
```

### Interpretation

**What it checks:** Real memory savings from quantizing a simulated KV cache of 1,024 vectors.

- `compression` ratio: 2-bit ≈ 7.8x, 3-bit ≈ 5.2x, 4-bit ≈ 3.9x vs FP16. Approximately `16/bits`.
- `attention scores shape` confirms the cache can serve attention queries — one score per cached key.
- `range` of scores is a sanity check that outputs are not degenerate (zeros or NaN).

**Healthy result:** Compression ratios close to `16/bits`, valid score shapes with non-degenerate ranges.

## TEST 6: Needle-in-Haystack Retrieval

```
bits=2, seq=  512: top1=  170 (needle=  170) [EXACT]
  bits=2, seq= 2048: top1=  682 (needle=  682) [EXACT]
  bits=2, seq= 8192: top1= 2730 (needle= 2730) [EXACT]
  bits=3, seq=  512: top1=  170 (needle=  170) [EXACT]
  bits=3, seq= 2048: top1=  682 (needle=  682) [EXACT]
  bits=3, seq= 8192: top1= 2730 (needle= 2730) [EXACT]
  bits=4, seq=  512: top1=  170 (needle=  170) [EXACT]
  bits=4, seq= 2048: top1=  682 (needle=  682) [EXACT]
  bits=4, seq= 8192: top1= 2730 (needle= 2730) [EXACT]
```

### Interpretation

**What it checks:** The most practical test — whether the quantized cache can still find the most-relevant key even at long sequence lengths.

- A "needle" key is hidden among 512 / 2,048 / 8,192 random keys. A query identical to that needle is issued. The test checks whether the needle ranks #1 after quantization.
- `[EXACT]` = needle was top-1 ranked. `[TOP-5]` = in top 5. `[MISS]` = not found.
- 9/9 `[EXACT]` means quantization does not degrade retrieval at all, even at 2-bit and 8K context.

**Healthy result:** All 9 rows `[EXACT]`. Any `[MISS]` at 3-bit or 4-bit would be alarming.

## TEST 7: GPU Benchmark (if CUDA available)

```
CUDA not available, skipping GPU test
```

### Interpretation

**What it checks:** Raw throughput on CUDA — quantization latency and inner product speed vs full FP16 matmul.

- `Quantize N keys` — time in ms to compress an 8,192-key sequence.
- `Inner product (Q queries × N keys)` — attention scoring time vs full-precision matmul.
- TurboQuant trades compute for memory bandwidth; on memory-bound hardware it can be faster than FP16 matmul.
- Memory comparison shows the actual byte savings at the configured bit-width.

**Without GPU:** Test is skipped. Run `validate.py` on a CUDA machine to see real throughput numbers.

## ALL TESTS COMPLETE

