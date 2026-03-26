# Section 8: Code Walkthrough -- Reading the PyTorch Implementation

The previous sections developed the theory; this section grounds it in code. We will walk through the four core files in the TurboQuant repository, tracing how each mathematical concept maps to a specific class, method, or line of code. The goal is not to explain every line, but to give you a reading map so you can navigate the codebase confidently and understand the design decisions behind the implementation.

## File Structure

The codebase is compact -- four source files and two test/validation files:

```
turboquant/
  __init__.py           -- Package exports
  lloyd_max.py          -- Lloyd-Max codebook solver (foundation)
  turboquant.py         -- Core: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
  compressors.py        -- V2 compressors for real model integration
  validate.py           -- Real model validation (Qwen 2.5 3B)
  test_turboquant.py    -- Mathematical validation tests (7 tests)
```

The dependency chain flows bottom-up: `lloyd_max.py` is the foundation, `turboquant.py` builds on it, and `compressors.py` wraps everything for production use.

## lloyd_max.py: The Foundation

This file contains the Lloyd-Max codebook solver -- the algorithm that finds the optimal scalar quantizer for a known distribution (Section 4).

**`beta_pdf(x, d)`** (lines 18-23) implements the exact distribution of each coordinate after random rotation of a d-dimensional unit vector. **`gaussian_approx_pdf(x, d)`** (lines 26-29) provides the N(0, 1/d) Gaussian approximation used in practice.

**`solve_lloyd_max(d, bits, pdf_func)`** (lines 32-86) is the iterative solver. The key design choice is on lines 52-53: centroids are initialized uniformly in [-3.5 * sigma, 3.5 * sigma] where sigma = 1/sqrt(d). This range covers 99.95% of the Gaussian mass, ensuring the solver starts with good coverage. The iteration loop (line 55) alternates between computing boundaries as midpoints (line 57) and updating centroids as conditional expectations via `scipy.integrate.quad` (lines 62-71). Convergence is checked against a tolerance of 1e-10 (line 74) -- this is tight, reflecting the fact that the codebook is computed once and reused forever.

**`LloydMaxCodebook`** (lines 107-131) wraps the solver into a PyTorch module. Two methods matter:

- **`quantize(x)`** (line 117): Maps each scalar to its nearest centroid index via `argmin` over absolute differences with all centroids. This is a brute-force nearest-neighbor search, which is fine because the codebook has at most 2^4 = 16 entries.
- **`dequantize(indices)`** (line 123): Looks up centroid values from indices -- a simple table lookup.

The class also precomputes `self.distortion` -- the expected MSE per coordinate, computed analytically by `compute_expected_distortion()` (lines 89-104). This number is used in tests to verify that empirical distortion matches theory.

## turboquant.py: The Core Engine

This file contains the three main classes that implement TurboQuant's two-stage pipeline.

### generate_rotation_matrix() (lines 18-33)

Generates a Haar-distributed random orthogonal matrix via QR decomposition of a Gaussian matrix, with the sign correction on the diagonal of R to ensure true uniform distribution over the orthogonal group (covered in detail in Section 4). The seed parameter ensures reproducibility -- the same seed always produces the same rotation matrix.

### generate_qjl_matrix() (lines 36-48)

Generates the random Gaussian projection matrix S for QJL. The default is m = d (same number of projections as input dimensions). Like the rotation matrix, it is seeded for reproducibility. The matrix entries are i.i.d. N(0,1) -- no normalization or orthogonalization is applied, because the JL guarantee works with raw Gaussian matrices.

### TurboQuantMSE (lines 51-100)

This class implements Stage 1: random rotation + Lloyd-Max per-coordinate quantization. It is the MSE-only quantizer used for value vectors.

**Constructor** (lines 57-69): The key line is `self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))` (line 64). The rotation matrix is stored as a buffer -- persistent, device-aware, but not a learnable parameter. The `LloydMaxCodebook` is created with the specified dimension and bit width.

**`rotate(x)`** and **`unrotate(y)`** (lines 71-78): Rotation is `x @ Pi^T` and un-rotation is `y @ Pi`. Since Pi is orthogonal, Pi^(-1) = Pi^T, so these are exact inverses. Note the convention: x is a row vector (batch of row vectors), so rotation multiplies on the right by Pi^T rather than on the left by Pi.

**`quantize(x)`** (line 80) and **`dequantize(indices)`** (line 88) contain the actual work. `quantize()` rotates the input and finds the nearest centroid index per coordinate. `dequantize()` looks up centroid values and unrotates back to the original space. **`forward(x)`** (line 93) simply delegates to both: `indices = self.quantize(x)` followed by `x_hat = self.dequantize(indices)`. The conceptual pipeline across these methods is four steps:

1. Rotate: `y = x @ Pi^T`
2. Quantize: find nearest centroid index for each coordinate
3. Dequantize: look up centroid values
4. Unrotate: `x_hat = centroids @ Pi`

The method returns both the reconstructed vector `x_hat` and the quantization indices. The indices are what gets stored; the reconstruction is computed on the fly when needed.

### TurboQuantProd (lines 103-196)

This class implements the full two-stage pipeline: MSE quantization (Stage 1) + QJL error correction (Stage 2). It is the quantizer used for key vectors, where unbiased inner products are essential.

**Constructor** (lines 112-132): Two critical design choices:
1. `self.mse_bits = max(bits - 1, 1)` (line 124): Stage 1 uses (b-1) bits for MSE quantization, reserving 1 bit for the QJL signs. The `max(..., 1)` guard prevents 0-bit MSE quantization if someone passes bits=1. At 3-bit total, Stage 1 uses 2 bits (4 centroids) and Stage 2 uses 1 bit (sign). This is why the total bit budget is b bits per coordinate.
2. The QJL matrix S uses `seed + 1` (line 132): This ensures S is independent of the rotation matrix Pi (which uses `seed`). Statistical independence between the rotation and the projection is important for the unbiasedness guarantee.

**`quantize(x)`** (lines 134-159): Returns a dictionary with three components:
- `"mse_indices"`: The Lloyd-Max codebook indices from Stage 1 (stored as uint8)
- `"qjl_signs"`: The sign-quantized projections of the residual (stored as {-1, +1})
- `"residual_norm"`: The L2 norm of the residual r = x - x_hat (stored as fp16)

The residual computation (lines 141-143) is straightforward: `residual = x - x_hat` followed by `residual_norm = torch.norm(residual, dim=-1, keepdim=True)`. The QJL projection (lines 145-148) multiplies the residual by S^T and takes the sign, with zeros mapped to +1.

**`inner_product(y, compressed)`** (lines 165-192): This is the method we dissected in Section 6 -- the asymmetric estimator. The query y is projected through S but not quantized; the inner product is computed as term1 (MSE reconstruction dot product) + term2 (QJL correction with `correction_scale = sqrt(pi/2) / m`).

**`dequantize(compressed)`** (lines 161-163): Only reconstructs the MSE component -- it does not use the QJL signs. This is used when you need an approximate vector back (e.g., for value reconstruction), not for inner products.

## compressors.py: Production Integration

This file bridges the gap between the core quantizers and real transformer models. It handles 4D tensor shapes (batch, heads, sequence, head_dim), explicit normalization, and the asymmetric attention score computation.

### TurboQuantCompressorV2 (lines 25-158)

The production key compressor. Two methods define its interface:

**`compress(states)`** (lines 83-120): Takes key states of shape (B, H, S, D), flattens to (B*H*S, D), and runs the full pipeline:
1. **Normalize**: Divide each vector by its L2 norm (line 95-96). This is an explicit step that `TurboQuantProd` handles implicitly -- the V2 compressor stores vector norms separately.
2. **Rotate**: Multiply by Pi^T (line 99).
3. **Quantize**: Find nearest Lloyd-Max centroid indices (lines 100-101).
4. **Reconstruct k_mse**: Dequantize, unrotate, and rescale by the original norm (line 104-105). This gives the MSE reconstruction in the original (un-normalized) space.
5. **Compute residual**: `residual = flat - k_mse` (line 108).
6. **QJL projection**: Project residual through S and take signs (lines 110-111). Signs are stored as int8 ({-1, +1}) rather than packed bits -- a convenience that uses 8x more memory than optimal but simplifies the PyTorch implementation.

The compressed output is a dictionary: `{k_mse (fp16), qjl_signs (int8), residual_norm (fp16)}`, reshaped back to 4D.

**`asymmetric_attention_scores(queries, compressed)`** (lines 122-158): Computes attention scores from the formula in Section 6. Term 1 is a standard `torch.matmul` between queries and k_mse^T. Term 2 projects queries through S, computes inner products with the sign vectors, and scales by `sqrt(pi/2) / m * residual_norm`. The result is term1 + term2 -- full attention scores without decompressing the keys.

### TurboQuantCompressorMSE (lines 161-221)

The value compressor. Simpler than V2 because values do not need QJL correction. It stores quantization indices (uint8) and vector norms (fp16), and provides a `decompress()` method that reconstructs value vectors by dequantizing, unrotating, and rescaling. No asymmetric attention method here -- values are fully decompressed before being used in the softmax-weighted sum.

### Key Design Choice: Keys vs. Values

The split between `TurboQuantCompressorV2` (keys) and `TurboQuantCompressorMSE` (values) reflects the different requirements derived in Section 6:

- **Keys** need unbiased inner products for attention scores, so they use the full two-stage pipeline with QJL correction. The trade-off: keys store k_mse (fp16) + signs (int8) + residual norm (fp16).
- **Values** need good MSE reconstruction for the weighted average, so they use Stage 1 only. The trade-off: values store indices (uint8) + vector norms (fp16) -- less memory per vector.

This asymmetry is a principled design decision, not a shortcut. It allocates the QJL overhead (sign bits + residual norms) only where it is mathematically necessary.

## Key Takeaways

- **The codebase is four files deep**: `lloyd_max.py` (codebook solver) -> `turboquant.py` (core quantizers) -> `compressors.py` (production wrappers).
- **`LloydMaxCodebook`** precomputes the optimal codebook once; `quantize()` and `dequantize()` are simple nearest-neighbor and table-lookup operations.
- **`TurboQuantMSE`** stores the rotation matrix as a buffer and implements rotate/quantize/dequantize/unrotate as its forward pass.
- **`TurboQuantProd`** uses (b-1) bits for MSE + 1 bit for QJL signs, with `inner_product()` implementing the asymmetric estimator.
- **`TurboQuantCompressorV2`** handles the full pipeline for real models: normalize, rotate, quantize, compute k_mse, compute residual, project and sign-quantize, then compute asymmetric attention scores directly from compressed data.
