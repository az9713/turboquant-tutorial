# TurboQuant Research Notes

Compiled from: Google Research blog, YouTube transcript (Tonbi Studio), TurboQuant paper (ICLR 2026), PolarQuant paper (AISTATS 2026), and the complete codebase implementation.

---

## Part A: The KV Cache Problem

### How Transformer Self-Attention Works

Transformer models compute self-attention using three projections of each input token:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I carry?"

The attention mechanism computes:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

where `d_k` is the head dimension. For each new token, the model must compute its query vector and compare it against ALL previous keys to determine which past tokens to attend to, then retrieve a weighted combination of their values.

### Why the KV Cache Exists

During autoregressive generation (producing one token at a time), recomputing Q, K, V for all previous tokens at every step would be O(n^2) in computation. The **KV cache** stores the K and V vectors from all previous tokens so they only need to be computed once. Each new token only computes its own Q, K, V projections, then uses the cached K and V for attention.

### How KV Cache Memory Grows

KV cache memory scales as:

```
Memory = layers x heads x seq_len x head_dim x 2 (K and V) x bytes_per_element
```

For the **Qwen 2.5 3B** model used in the transcript's validation:
- 36 layers (from transcript: "across all 36 layers")
- 2 KV heads per layer (from transcript: "two key value heads per layer")
- head_dim = 128 (standard for this model class; confirmed in code: `head_dim = cache.layers[0].keys.shape[-1]`)
- At fp16 (2 bytes per element)

For 8K context:
```
36 x 2 x 8192 x 128 x 2 x 2 bytes = 289 MB
```

This matches the transcript exactly: "the original was 289 megabytes" (transcript line ~704).

### Why This Is the Bottleneck

The transcript explains the practical impact (lines ~598-608):
- 8K context at fp16 needs ~289 MB just for the KV cache
- The Qwen 2.5 3B model in 4-bit weights takes ~2 GB of VRAM
- On a 12 GB RTX 3060, the KV cache quickly becomes the dominant memory consumer
- At longer contexts, "8K context that needed 10GB now fits in under 2GB" (transcript line ~23)
- The implication: compressing the KV cache from 289 MB to 58 MB (at 3-bit) means "the difference between fitting 8,000 context and fitting 40,000 with the same model" (transcript lines ~730-732)

### The Core Constraint

The blog states: "High-dimensional vectors are incredibly powerful, but they also consume vast amounts of memory, leading to bottlenecks in the key-value cache" (blog line ~94). The KV cache is a "high-speed digital cheat sheet" that grows linearly with conversation length, and on consumer hardware, it is often the binding constraint on context length.

---

## Part B: Quantization Fundamentals

### What Quantization Is

Quantization maps continuous (or high-precision) values to a smaller set of discrete symbols. The transcript's analogy (lines ~176-199): "Think of this as like compressing a photo. A raw photo off your camera might be 50 megabytes, but when you save it as a JPEG, it drops to 2 megabytes. And honestly, you can barely tell the difference."

In the AI context: instead of storing each number with 16 or 32 bits of precision, quantization rounds to 2-4 bits per number.

### Scalar vs. Vector Quantization

- **Scalar quantization**: Quantize each coordinate independently. Simple but suboptimal because it ignores correlations between coordinates.
- **Vector quantization (VQ)**: Quantize groups of coordinates together using a codebook of representative vectors. Better compression but exponential codebook size in dimension.
- **Product quantization (PQ)**: A compromise — split the d-dimensional vector into M sub-vectors and apply VQ to each independently.

### The MSE Objective

The goal of quantization is to minimize the **Mean Squared Error** (MSE):

```
minimize E[||x - Q(x)||^2]
```

where Q(x) is the quantized (reconstructed) version of x. This measures the average distortion introduced by compression.

### The Codebook

A codebook is a set of representative values (centroids) `{c_1, c_2, ..., c_K}` where K = 2^b for b bits. Each input value is mapped to the nearest centroid, and only the index (b bits) is stored instead of the full value.

### Why Naive Rounding Is Suboptimal

Uniform quantization (evenly-spaced levels) is suboptimal when the data distribution is non-uniform. If most values cluster near zero (as in Gaussian distributions), uniform quantization wastes levels in sparsely-populated regions and provides insufficient resolution where density is high.

### Why Activations Follow Gaussian Distributions

After normalization layers (LayerNorm, RMSNorm) in transformers, activation vectors tend to have approximately Gaussian-distributed coordinates. More precisely, after random rotation of a unit-norm vector in d dimensions, each coordinate follows a Beta-type distribution that is well-approximated by N(0, 1/d) for d >= 64. This is stated in the lloyd_max.py docstring (lines 1-10): "For practical dimensions (d >= 64), this is well-approximated by N(0, 1/d)."

---

## Part C: The Memory Overhead Problem (Classical VQ)

### The Scale Factor Problem

Classical vector quantization methods normalize each vector (or sub-vector) before quantizing. This normalization requires storing a **scale factor** (the original norm or standard deviation) for each block of data. These scale factors are stored in full precision (fp16 or fp32).

### Product Quantization Overhead

Product Quantization (PQ) splits a d-dimensional vector into M sub-vectors of d/M dimensions each. Each sub-vector requires:
1. Its own codebook indices (the compressed data)
2. A normalization constant / scale factor (stored at full precision)

The per-sub-vector scale factor adds 16 or 32 bits of overhead per sub-vector. If M is large (small sub-vectors), the overhead per coordinate becomes 1-2 extra bits, which is significant when the target is only 2-4 bits per coordinate.

### The Blog's Key Statement

The blog (line ~96) states: "traditional vector quantization usually introduces its own 'memory overhead' as most methods require calculating and storing (in full precision) quantization constants for every small block of data. This overhead can add 1 or 2 extra bits per number, **partially defeating the purpose** of vector quantization."

### TurboQuant's Key Insight

TurboQuant addresses this by quantizing **normalized** vectors using a **fixed codebook** (determined by the known distribution, not the data), and handling the scale information through the QJL residual norm — a single fp16 value per entire vector, not per sub-block. This achieves "zero memory overhead" in the sense that the overhead does not scale with d.

---

## Part D: Lloyd-Max Scalar Quantizer

**Source file**: `lloyd_max.py`

### What Lloyd-Max Is

The Lloyd-Max algorithm finds the **optimal scalar quantizer** for a known probability distribution. It is the 1-dimensional case of k-means clustering, applied to the probability density function rather than to samples.

### Optimality Conditions

The Lloyd-Max quantizer satisfies two conditions simultaneously:

1. **Nearest-neighbor encoding**: Each input value is mapped to the nearest centroid (boundary = midpoint between adjacent centroids)
2. **Centroid condition**: Each centroid equals the conditional expectation of the distribution within its Voronoi cell:
   ```
   c_i = E[X | X in cell_i] = integral(x * f(x) dx, a_i, a_{i+1}) / integral(f(x) dx, a_i, a_{i+1})
   ```

### The Iterative Procedure

**`solve_lloyd_max()` function (lloyd_max.py, lines 32-86)**:

1. **Initialize** centroids uniformly in [-3.5*sigma, 3.5*sigma] where sigma = 1/sqrt(d) (line 52-53)
2. **Iterate** up to 200 times (line 55):
   a. Compute boundaries as midpoints between adjacent centroids (line 57)
   b. Update each centroid as the conditional expectation E[X | X in partition_i] using numerical integration (`scipy.integrate.quad`) (lines 62-71)
   c. Check convergence: stop if max centroid shift < 1e-10 (lines 74-78)
3. Return final centroids and boundaries as torch tensors (lines 83-86)

### The Distribution: N(0, 1/d)

The exact distribution of each coordinate after random rotation of a d-dimensional unit vector is a Beta-type distribution (lloyd_max.py, lines 6-8):

```
f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
```

supported on [-1, 1]. The `beta_pdf()` function (lines 18-23) implements this exactly.

For d >= 64, this is well-approximated by the Gaussian N(0, 1/d). The `gaussian_approx_pdf()` function (lines 26-29) implements this approximation:

```python
sigma2 = 1.0 / d
return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))
```

### The Symmetry Property

The codebook is **symmetric around zero** because the underlying distribution (both the exact Beta and the Gaussian approximation) is symmetric. This is verified in `test_turboquant.py` (lines 31-35):

```python
cb = LloydMaxCodebook(128, 3)
centroid_sum = cb.centroids.sum().abs().item()
assert centroid_sum < 0.01, "Centroids should be symmetric!"
```

The transcript confirms (lines ~440-455): "the symmetry check was at zero so it means the code book which is the dictionary the compressor uses to translate numbers needs to be perfectly balanced so it's symmetric around zero. If it wasn't the compression would favor certain values and that would introduce a lot of errors."

### The LloydMaxCodebook Class (lines 107-131)

- **Constructor** (`__init__`, line 110): Takes dimension d and bits, precomputes centroids, boundaries, and expected distortion
- **`quantize()`** (line 117): Maps values to nearest centroid indices using `argmin` over absolute differences
- **`dequantize()`** (line 123): Looks up centroid values from indices
- Key detail: `self.distortion` stores the expected distortion per coordinate, computed analytically by `compute_expected_distortion()` (lines 89-104)

### Empirical MSE Within Theoretical Bounds

From the transcript (lines ~456-468): "for MSE distortion we found that empirical MSE stayed well within the theoretical bounds... the paper says that the error will be no worse than X. Our actual error came well under that."

The theoretical bound from the test (test_turboquant.py, line 63):
```python
theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))
```

This comes from the paper's Theorem on near-optimal distortion rates.

---

## Part E: The Random Rotation Trick

**Source file**: `turboquant.py`

### Why Random Rotation Helps

When a vector x has correlated coordinates or non-uniform coordinate variances, per-coordinate scalar quantization is suboptimal. Applying a random orthogonal rotation makes all coordinates approximately i.i.d. (independent and identically distributed), which makes per-coordinate quantization near-optimal.

**Key property**: If x ~ N(0, sigma^2 * I), then Pi @ x ~ N(0, sigma^2 * I) for any orthogonal matrix Pi. The rotation preserves the distribution while randomizing which directions the coordinates represent.

For unit-norm vectors: after rotation, each coordinate follows approximately N(0, 1/d), which is exactly the distribution the Lloyd-Max codebook is optimized for.

### Haar-Distributed Random Orthogonal Matrix

The `generate_rotation_matrix()` function (turboquant.py, lines 18-33) generates a Haar-distributed (uniformly random) orthogonal matrix:

```python
def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)       # Random Gaussian matrix
    Q, R = torch.linalg.qr(G)                   # QR decomposition
    diag_sign = torch.sign(torch.diag(R))        # Sign correction
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)               # Ensure det(Q) = +1
    return Q.to(device)
```

**Step-by-step**:
1. Generate a d x d matrix G with i.i.d. N(0,1) entries
2. Compute QR decomposition: G = Q @ R
3. **Sign correction**: Multiply columns of Q by sign(diag(R)). This is necessary because QR decomposition has a sign ambiguity — without this correction, the resulting Q would not be uniformly distributed over the orthogonal group. The sign correction ensures Haar measure.

### Rotation Is Stored, Not Recomputed

In `TurboQuantMSE.__init__()` (turboquant.py, line 64):
```python
self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))
```

The rotation matrix is registered as a PyTorch buffer, meaning it is:
- Computed once at initialization
- Stored with the module (moves with `.to(device)`, saved/loaded with state_dict)
- NOT a learnable parameter (no gradients)
- The same matrix is used for both quantization and dequantization

### How Rotation Is Applied

In `TurboQuantMSE` (turboquant.py, lines 71-78):
```python
def rotate(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.Pi.T    # y = x @ Pi^T (equivalent to Pi @ x for each row vector)

def unrotate(self, y: torch.Tensor) -> torch.Tensor:
    return y @ self.Pi       # x = y @ Pi (inverse of orthogonal = transpose)
```

Since Pi is orthogonal (Pi @ Pi^T = I), the inverse rotation is simply the transpose.

---

## Part F: PolarQuant -- Polar Coordinate Quantization

**Source**: Blog post (lines 110-114), paper (AISTATS 2026)

### The Key Innovation

PolarQuant converts vectors from **Cartesian coordinates** (x, y, z, ...) to **polar coordinates** (radius + angles). This fundamental change of representation eliminates the need for stored scale factors.

### Why Polar Representation Eliminates Scale Factors

In Cartesian quantization, the quantization boundaries depend on the scale (norm) of the vector. Different vectors with different norms need different boundary sets, requiring stored scale factors.

In polar coordinates:
- The **radius** captures the scale (norm) of the vector — a single number
- The **angles** describe the direction, independent of scale
- The angular distribution is **concentrated and predictable** — it follows a known distribution regardless of the input data's scale
- The quantization grid for angles is **fixed** — boundaries don't change with data scale

### The Blog's Analogy

Blog line ~111: "This is comparable to replacing 'Go 3 blocks East, 4 blocks North' with 'Go 5 blocks total at a 37-degree angle'."

In Cartesian: the "3" and "4" depend on scale. In polar: the angle (37 degrees) is scale-invariant, and the radius (5) is a single number.

### The Recursive Structure

Blog lines ~113: "The mechanism begins by grouping pairs of coordinates from a d-dimensional vector and mapping them onto a polar coordinate system. Radii are then gathered in pairs for recursive polar transformations — a process that repeats until the data is distilled into a single final radius and a collection of descriptive angles."

Concretely for a d-dimensional vector:
1. Group coordinates into d/2 pairs: (x_1, x_2), (x_3, x_4), ...
2. Convert each pair to polar: (r_i, theta_i) where r_i = sqrt(x_{2i-1}^2 + x_{2i}^2)
3. Now we have d/2 radii. Group those into pairs and convert to polar again.
4. Recurse until we have: **one final radius** (the original vector's norm) + **log_2(d) layers of angles**

### Zero Memory Overhead

The angular grid is fixed and predictable (determined by the known distribution), so no per-block normalization constants need to be stored. The only "overhead" is the single final radius, which is O(1) regardless of dimension d.

### Relationship to TurboQuant

TurboQuant uses the **random rotation + Lloyd-Max** approach (which achieves similar benefits to PolarQuant via a different mechanism) rather than explicitly implementing PolarQuant's recursive polar conversion. The rotation trick makes coordinates i.i.d. so a fixed codebook works, achieving the same "zero overhead" property. PolarQuant is the theoretical foundation; TurboQuant's implementation uses the rotation approach as a practical equivalent.

---

## Part G: QJL -- Quantized Johnson-Lindenstrauss

**Source file**: `turboquant.py` (lines 36-48 for matrix generation, lines 165-192 for the estimator)

### The Johnson-Lindenstrauss Lemma

The JL lemma states that random projections preserve distances and inner products in expectation. For a random matrix S in R^(m x d) with i.i.d. N(0,1) entries:

```
E[<Sx, Sy> / m] = <x, y>
```

This means we can estimate inner products from lower-dimensional projections.

### The QJL Trick: 1-Bit Projection

QJL takes this further: instead of storing the full projected vector Sx, store only the **signs**:

```
sign(Sx) in {-1, +1}^m
```

This requires only 1 bit per projected dimension. The key mathematical result is that sign-based projections still preserve inner product information (up to scaling).

### The Asymmetric Estimator (The Core Formula)

**This is the centerpiece of TurboQuant.** For a compressed key k and full-precision query q, the asymmetric estimator is:

```
<q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2) / m * <Sq, sign(Sr_k)>
```

where:
- `k_mse` = Stage 1 MSE reconstruction of k
- `r_k = k - k_mse` = the residual (error from Stage 1)
- `S` = random Gaussian projection matrix (m x d)
- `||r_k||` = L2 norm of the residual
- `m` = projection dimension (typically m = d)

**Why this is asymmetric**: The query q is used at full precision (projected through S but not sign-quantized). Only the key k is compressed. This is natural for attention: we compress keys once and reuse them for many queries.

### Implementation in Code

**`TurboQuantProd.inner_product()`** (turboquant.py, lines 165-192):

```python
def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
    # Term 1: inner product with MSE reconstruction
    x_mse = self.mse.dequantize(compressed["mse_indices"])
    term1 = (y * x_mse).sum(dim=-1)

    # Term 2: QJL correction
    y_projected = y @ self.S.T                              # Project query (NOT quantized)
    qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)

    m = self.qjl_dim
    correction_scale = math.sqrt(math.pi / 2) / m           # Line 189
    term2 = compressed["residual_norm"] * correction_scale * qjl_ip

    return term1 + term2
```

### Why This Estimator Is Unbiased

The estimator is **exactly unbiased**: E[estimator] = <q, k>.

**Proof sketch**:
1. Term 1: `<q, k_mse>` is deterministic (no randomness after quantization)
2. Term 2 corrects for the residual: `<q, r_k>` where r_k = k - k_mse
3. For the QJL part: `E[sign(S_i^T r_k)] = sqrt(2/pi) * r_k / ||r_k||` (well-known result for sign of Gaussian projection)
4. Therefore: `E[sqrt(pi/2)/m * <Sq, sign(Sr_k)> * ||r_k||] = <q, r_k>`
5. Combining: `E[term1 + term2] = <q, k_mse> + <q, r_k> = <q, k_mse + r_k> = <q, k>`

The `sqrt(pi/2)` factor is the correction for the distortion introduced by taking signs.

### The correction_scale

In turboquant.py line 189:
```python
correction_scale = math.sqrt(math.pi / 2) / m
```

And in compressors.py line 154:
```python
correction_scale = math.sqrt(math.pi / 2) / m
```

This factor appears because `E[|Z|] = sqrt(2/pi)` for Z ~ N(0,1), and the sign function introduces a factor of `sqrt(pi/2)` that must be compensated.

### The QJL Matrix Generation

**`generate_qjl_matrix()`** (turboquant.py, lines 36-48):
```python
def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)
```

The matrix S has shape (m, d) with i.i.d. N(0,1) entries. Default m = d (same dimensionality as input). The matrix is random but fixed (seeded), not data-dependent.

### The Blog's Description

Blog line ~107-108: "QJL uses a mathematical technique called the Johnson-Lindenstrauss Transform to shrink complex, high-dimensional data while preserving the essential distances and relationships between data points. It reduces each resulting vector number to a single sign bit (+1 or -1). This algorithm essentially creates a high-speed shorthand that requires zero memory overhead."

---

## Part H: The Two-Stage TurboQuant Pipeline

**Source file**: `turboquant.py`, class `TurboQuantProd` (lines 103-196)

### Overview

TurboQuant combines two stages to achieve both low MSE and unbiased inner products:

- **Stage 1 (b-1 bits)**: Random rotation + Lloyd-Max per-coordinate quantization (MSE-optimal)
- **Stage 2 (1 bit)**: QJL on the residual from Stage 1 (bias correction)

### Stage 1: MSE Quantization

**`TurboQuantProd.quantize()`** (turboquant.py, lines 134-159):

```python
def quantize(self, x: torch.Tensor) -> dict:
    # Stage 1: MSE quantize
    x_hat, mse_indices = self.mse(x)          # Rotate, quantize, dequantize

    # Compute residual
    residual = x - x_hat
    residual_norm = torch.norm(residual, dim=-1, keepdim=True)

    # Stage 2: QJL - project residual and take sign
    projected = residual @ self.S.T            # (batch, qjl_dim)
    qjl_signs = torch.sign(projected)          # (batch, qjl_dim)
    qjl_signs[qjl_signs == 0] = 1.0           # map zeros to +1
```

Step-by-step for Stage 1:
1. **Rotate**: y = x @ Pi^T (turboquant.py line 82)
2. **Quantize**: Find nearest Lloyd-Max centroid for each coordinate (line 84-85)
3. **Dequantize**: Look up centroids, then unrotate: x_hat = centroids[indices] @ Pi (lines 90-91)

### Stage 2: QJL on Residuals

1. **Compute residual**: r = x - x_hat (the error from Stage 1)
2. **Project**: projected = r @ S^T (multiply by random Gaussian matrix)
3. **Sign-quantize**: signs = sign(projected), stored as {-1, +1}
4. **Store norm**: ||r|| stored as a single fp16 value per vector

### Total Storage Per Vector

```
Stage 1: (b-1) * d bits        — MSE codebook indices
Stage 2: d bits                 — QJL sign bits (1 bit per projected dim, m=d)
         + 16 bits              — residual norm (fp16 scalar)
-----------------------------------------
Total:   b * d + 16 bits        — effectively b bits per coordinate
```

The "16 bits for residual norm" is amortized over d coordinates, adding only 16/d bits per coordinate — negligible for d >= 64.

### Why Combining Both Stages Eliminates Bias

From the transcript (lines ~510-531):
- "Stage one alone can result in biased findings"
- "with stage one and QJL that was where we were able to secure an unbiased finding"
- Analogy: "imagine compressing a photo and the colors shift slightly blue. Stage one is the compression. Stage two is the automatic color correction that perfectly undoes the shift and at the cost of just one extra bit per number."

Mathematically: Stage 1 (MSE quantization) introduces systematic error because `<q, k_mse>` != `<q, k>` in expectation (the quantization distortion is correlated with the signal). Stage 2 (QJL) adds a correction term that is exactly unbiased, making the combined estimator unbiased.

### The test_turboquant.py Verification

**Test 3** (test_turboquant.py, lines 74-113) verifies unbiasedness:
```python
bias = (estimated_ip - true_ip).mean().item()
```
The bias is near zero for all bit widths (2, 3, 4).

**Test 4** (lines 116-144) shows the contrast — MSE-only inner products ARE biased:
```python
bias = (mse_ip - true_ip).mean().item()
print(f"  bits={bits}: bias={bias:+.6f} (MSE-only is biased, QJL fixes this)")
```

This demonstrates that QJL is essential for unbiased attention scores.

### The TurboQuantProd Class Structure

- **`__init__`** (lines 112-132): Creates the MSE quantizer with (bits-1) bits and the QJL matrix S. Uses `seed+1` for S to ensure independence from the rotation matrix.
- **`quantize()`** (lines 134-159): Returns dict with `mse_indices`, `qjl_signs`, `residual_norm`
- **`dequantize()`** (line 161-163): Only reconstructs the MSE component (for getting back approximate vectors)
- **`inner_product()`** (lines 165-192): The asymmetric estimator — the key innovation
- **`forward()`** (line 194-196): Alias for `quantize()`

---

## Part I: Asymmetric Attention

**Source file**: `compressors.py`

### The V2 Compressor: TurboQuantCompressorV2

This is the production-oriented compressor (compressors.py, lines 25-158) designed for actual transformer KV cache integration. Unlike `TurboQuantProd` (which operates on flat batches), this handles the 4D shape (batch, heads, seq, head_dim) directly.

### Compression: compress() Method (lines 83-120)

```python
def compress(self, states: torch.Tensor) -> dict:
    B, H, S, D = states.shape
    flat = states.reshape(-1, D).float()

    # Store original norms
    vec_norms = torch.norm(flat, dim=-1, keepdim=True)
    flat_norm = flat / (vec_norms + 1e-8)

    # Rotate and quantize
    rotated = flat_norm @ self.Pi.T
    diffs = rotated.unsqueeze(-1) - self.centroids
    indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

    # MSE reconstruction in original space
    reconstructed_rotated = self.centroids[indices.long()]
    k_mse = (reconstructed_rotated @ self.Pi) * vec_norms    # Back to original scale

    # Residual and QJL
    residual = flat - k_mse
    residual_norm = torch.norm(residual, dim=-1)
    projected = residual @ self.S.T
    signs = (projected >= 0).to(torch.int8) * 2 - 1          # {-1, +1} as int8
```

**Key difference from TurboQuantProd**: The V2 compressor explicitly normalizes vectors (divides by norm) before rotation, then scales the MSE reconstruction back by the original norm. This handles vectors of varying magnitude. The residual and its norm are computed in the original (un-normalized) space.

### The Three Stored Components

The compressed representation (returned dict):
1. **`k_mse`**: MSE reconstruction in fp16, shape (B, H, S, D) — the dominant memory cost
2. **`qjl_signs`**: Sign bits as int8, shape (B, H, S, D) — 1 bit of useful info per element (stored as int8 for convenience)
3. **`residual_norm`**: L2 norm of residual in fp16, shape (B, H, S) — one scalar per vector

### Asymmetric Attention Scores: asymmetric_attention_scores() (lines 122-158)

```python
def asymmetric_attention_scores(self, queries, compressed):
    k_mse = compressed["k_mse"].float()       # (B, H, S_k, D)
    signs = compressed["qjl_signs"].float()     # (B, H, S_k, D)
    r_norm = compressed["residual_norm"].float() # (B, H, S_k)

    # Term 1: Q @ K_mse^T  (standard matmul)
    term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))

    # Term 2: QJL correction
    q_projected = torch.matmul(queries.float(), self.S.T)       # (B, H, S_q, D)
    qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1)) # (B, H, S_q, S_k)

    m = self.S.shape[0]
    correction_scale = math.sqrt(math.pi / 2) / m
    term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)

    return term1 + term2
```

**The formula implemented**:
```
scores[i,j] = Q[i] @ K_mse[j]^T + sqrt(pi/2)/m * (Q[i] @ S^T) @ signs[j]^T * ||r_j||
```

This computes attention scores **without ever decompressing the keys**. The queries are at full precision; only the keys are compressed. This is the "asymmetric" property — one side (keys) is quantized, the other (queries) is not.

### Why Values Use MSE-Only Compression

The `TurboQuantCompressorMSE` class (compressors.py, lines 161-221) is used for values. The docstring explains (compressors.py, lines 15-16):

"For values, we use MSE-only decompression since the weighted sum in softmax(scores) @ V averages out per-vector errors."

The attention output is: `output = softmax(scores) @ V`. Since V vectors are combined via a weighted average (softmax weights), individual per-vector errors in V reconstruction tend to cancel out, making the QJL correction unnecessary for values. This saves memory: values only need b bits per coordinate (no QJL signs or residual norms).

### Memory Layout

For keys (TurboQuantCompressorV2):
- `k_mse`: fp16 (16 bits per element) — this is the dominant cost
- `qjl_signs`: int8 (8 bits stored, 1 bit useful) — could be bit-packed for more savings
- `residual_norm`: fp16 (16 bits per vector, amortized)

For values (TurboQuantCompressorMSE):
- `indices`: uint8 (8 bits stored, b bits useful)
- `vec_norms`: fp16 (16 bits per vector, amortized)

### Memory Accounting in validate.py

The validation script (validate.py, lines 120-133) computes actual bit counts:
```python
k_bits = n_key_vecs * D * mse_bits       # MSE indices
k_bits += n_key_vecs * D * 1              # QJL sign bits
k_bits += n_key_vecs * 16                 # residual norms (fp16)
k_bits += n_key_vecs * 16                 # vector norms (fp16)

v_bits = n_key_vecs * D * bits            # Value indices (full bits, no QJL)
v_bits += n_key_vecs * 16                 # Value norms
```

---

## Part J: Empirical Results

### Synthetic KV Cache Compression Ratios (from test_turboquant.py, Test 5)

Using d=128, seq_len=1024 synthetic random vectors:
- **2-bit**: 7.76x compression
- **3-bit**: 5.22x compression
- **4-bit**: 3.94x compression

From the transcript (lines ~543-551): "We saw 7.76x compression at 2 bit, 5.22x compression at 3bit, and 3.94x at 4bit."

### Real Model Results (Qwen 2.5 3B on RTX 3060)

From the transcript (lines ~700-732):
- **Original KV cache**: 289 MB (fp16, 8K context)
- **4-bit compressed**: 76 MB (3.8x compression)
- **3-bit compressed**: 58 MB (5.0x compression)
- **2-bit compressed**: 40 MB (7.3x compression)

The model was Qwen 2.5 3B Instruct, loaded in 4-bit weights (bitsandbytes nf4), running on an NVIDIA RTX 3060 (12 GB VRAM). See validate.py line 18: `MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"`.

### Attention Fidelity

From the transcript (lines ~749-793):

**Cosine similarity** (compressed vs. uncompressed attention scores):
- **4-bit**: ~0.999 (99.9%)
- **3-bit**: 0.995 (99.5%)
- **2-bit**: lower (not exactly stated, but significantly worse)

**Top-1 match rate** (does the model attend to the same most-important token?):
- **3-bit**: 92% across all 36 layers and 2 KV heads per layer
- **2-bit**: 66% (transcript lines ~846-849: "the 66% top one match means the model would sometimes attend to different tokens which could change outputs")

**Top-5 match rate** (is the real top-1 in the compressed top-5?):
- **3-bit**: 92% (transcript line ~788)

### Needle-in-Haystack Results

From the transcript (lines ~555-582) and test_turboquant.py Test 6:
- **100% retrieval accuracy** (9/9 tests) across all bit widths (2, 3, 4)
- Test configuration: hide one specific key among 512, 2048, or 8192 others
- "zero information loss for retrieval" (transcript line ~566)

The validation test (validate.py) uses a more realistic setup: embed "The secret project code name is AURORA-7749" in a long document, compress the KV cache, and verify the model can still find it.

### Practical Implications

From the transcript (lines ~730-735):
- "That's the difference between fitting 8,000 context and fitting 40,000 with the same model"
- On 12 GB RTX 3060: 289 MB KV cache at fp16 becomes 58 MB at 3-bit
- Freed VRAM can support ~5x longer context

### The 3-Bit Sweet Spot

From the transcript (lines ~818-868):
- **3-bit is the practical sweet spot**: 5x compression with 99.5% attention fidelity
- **4-bit**: More conservative, 3.8x compression, near-perfect fidelity (~0.999)
- **2-bit**: Aggressive 7.3x compression but 66% top-1 match "means the model would sometimes attend to different tokens which could change outputs"
- The paper's "zero accuracy loss" claim is "reasonable at 3 to four bits" — "the attention patterns are so close to the original that generation quality would be nearly indistinguishable in practice"

---

## Part K: Key Theoretical Properties

### Zero Memory Overhead

TurboQuant achieves quantization without per-block scale factors:
- The rotation matrix Pi is fixed (one matrix for all vectors of a given dimension)
- The Lloyd-Max codebook is fixed (determined by the known N(0, 1/d) distribution)
- The only per-vector overhead is the residual norm ||r|| (1 fp16 = 16 bits per vector)
- This overhead is O(1) per vector regardless of dimension d, amounting to 16/d extra bits per coordinate

Blog line ~101: "TurboQuant is a compression method that achieves a high reduction in model size with zero accuracy loss."

The QJL projection matrix S is also fixed (random but precomputed), not data-dependent.

### Data-Oblivious

TurboQuant requires **no training or fine-tuning**:
- The rotation matrix Pi is random (not learned from data)
- The Lloyd-Max codebook is computed from the known distribution N(0, 1/d), not from data statistics
- The QJL matrix S is random Gaussian

From the transcript (lines ~276-280): "you don't need any kind of retraining on the models. You can just apply this to an existing model and it works."

From the blog (line ~136): "TurboQuant demonstrates a transformative shift in high-dimensional search. By setting a new benchmark for achievable speed, it delivers near-optimal distortion rates in a **data-oblivious** manner."

This is a crucial practical advantage: the compression can be applied to any model without calibration data or fine-tuning passes.

### Near-Optimal Distortion

The theoretical distortion bound for b-bit TurboQuant (from the test code, test_turboquant.py line 63):

**MSE distortion per coordinate**:
```
D_mse <= sqrt(3) * pi/2 * (1/4^b)
```

**Inner product distortion** (test_turboquant.py line 108):
```
D_prod <= sqrt(3) * pi^2/d * (1/4^b)
```

Note the inner product distortion scales as 1/d — it improves with higher dimensions. This means TurboQuant is especially effective for models with large head dimensions.

The blog (lines ~138-139): "These methods don't just work well in real-world applications; they are provably efficient and operate near theoretical lower bounds."

### Variance O(1/d)

The QJL estimator's variance decreases as 1/d (the head dimension). This is stated in the compressors.py docstring (lines 11-12): "This is unbiased with variance O(1/d), even though k_mse itself has high per-vector error."

**Why this matters**: For typical head dimensions of 64-128 (or even 256 in newer models), the variance of the inner product estimator is very small. At d=128, the standard deviation of the estimator error scales as 1/sqrt(128) ≈ 0.088 of the signal.

This means:
- Larger head dimensions → more accurate compression
- The method scales well with modern architectures that use larger heads
- Even at aggressive 2-bit quantization, the estimator variance is controlled

### Unbiasedness: The Critical Property

The most important theoretical property is **unbiasedness** of the inner product estimator. This is critical for attention because:

1. Attention scores go through softmax, which is sensitive to relative magnitudes
2. A biased estimator would systematically shift which tokens get attended to
3. An unbiased estimator means errors are random, and on average, the attention pattern is correct

From the transcript (lines ~482-500): "when AI picks which word to say next, it compares the vectors using inner products. If a compression makes those comparisons consistently wrong in one direction, the AI is going to give bad answers. So the bias being very close to zero means the compression doesn't skew the AI's decision-making."

Test 3 in test_turboquant.py verifies this empirically: bias is near-zero for all tested configurations.

Test 4 shows the contrast: MSE-only inner products ARE biased, proving that QJL correction is necessary.

---

## Appendix: Code Architecture Summary

### File Structure

```
turboquant/
  __init__.py           — Package exports
  lloyd_max.py          — Lloyd-Max codebook solver (foundation)
  turboquant.py         — Core TurboQuant: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
  compressors.py        — V2 compressors for real model integration
  validate.py           — Real model validation (Qwen 2.5 3B)
  test_turboquant.py    — Mathematical validation tests (7 tests)
```

### Key Classes and Their Roles

| Class | File | Purpose |
|-------|------|---------|
| `LloydMaxCodebook` | lloyd_max.py | Optimal scalar quantizer for N(0, 1/d) |
| `TurboQuantMSE` | turboquant.py | Stage 1: rotation + per-coordinate quantization |
| `TurboQuantProd` | turboquant.py | Stage 1+2: MSE + QJL for unbiased inner products |
| `TurboQuantKVCache` | turboquant.py | KV cache wrapper using TurboQuantProd/MSE |
| `TurboQuantCompressorV2` | compressors.py | Production key compressor with asymmetric attention |
| `TurboQuantCompressorMSE` | compressors.py | Production value compressor (MSE-only) |

### Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `solve_lloyd_max()` | lloyd_max.py | Iterative Lloyd-Max solver using scipy.integrate |
| `generate_rotation_matrix()` | turboquant.py | Haar-random orthogonal matrix via QR |
| `generate_qjl_matrix()` | turboquant.py | Random Gaussian projection matrix |
| `build_prompt()` | validate.py | Constructs needle-in-haystack test prompt |

### Data Flow for Key Compression (TurboQuantCompressorV2)

```
Input key: (B, H, S, D) fp16
  ↓ normalize (divide by ||k||)
  ↓ rotate (@ Pi^T)
  ↓ quantize (nearest Lloyd-Max centroid → uint8 indices)
  ↓ dequantize + unrotate → k_mse (fp16)
  ↓ residual = k - k_mse
  ↓ project residual through S → sign → qjl_signs (int8)
  ↓ ||residual|| → residual_norm (fp16)
Output: {k_mse, qjl_signs, residual_norm}
```

### Data Flow for Asymmetric Attention

```
Query Q: (B, H, S_q, D) fp16     Compressed K: {k_mse, signs, r_norm}
  ↓                                      ↓
  ├── Q @ k_mse^T ─────────────→ term1: (B, H, S_q, S_k)
  ├── Q @ S^T ──→ q_proj          ↓
  │               ↓               signs^T
  │               q_proj @ signs^T ──→ qjl_ip: (B, H, S_q, S_k)
  │                                    × r_norm × sqrt(pi/2)/m
  │                                    ──→ term2: (B, H, S_q, S_k)
  └── scores = term1 + term2 ──→ (B, H, S_q, S_k)
```

---

## Appendix: Key Numbers Quick Reference

| Metric | Value | Source |
|--------|-------|--------|
| KV cache original (8K ctx, Qwen 2.5 3B) | 289 MB | Transcript |
| KV cache 4-bit | 76 MB (3.8x) | Transcript |
| KV cache 3-bit | 58 MB (5.0x) | Transcript |
| KV cache 2-bit | 40 MB (7.3x) | Transcript |
| Cosine similarity at 3-bit | 0.995 | Transcript |
| Top-1 match at 3-bit | 92% | Transcript |
| Top-1 match at 2-bit | 66% | Transcript |
| Needle retrieval accuracy | 100% (9/9) | Transcript |
| Synthetic compression 2-bit | 7.76x | test_turboquant.py |
| Synthetic compression 3-bit | 5.22x | test_turboquant.py |
| Synthetic compression 4-bit | 3.94x | test_turboquant.py |
| Model layers (Qwen 2.5 3B) | 36 | Transcript |
| KV heads per layer | 2 | Transcript |
| Head dimension | 128 | Code |
| Lloyd-Max convergence iterations | 200 max | lloyd_max.py |
| Context expansion factor (3-bit) | 5x (8K → 40K) | Transcript |
| Google claimed speedup (H100) | up to 8x | Blog/Transcript |
| Correction scale formula | sqrt(pi/2) / m | turboquant.py line 189 |
