# TurboQuant Code Deep Dive
## Mapping the Implementation to the Theory

*Companion to [`turboquant_tutorial.md`](turboquant_tutorial.md) — read the tutorial first, then use this document to map every abstract concept to its concrete lines of code.*

---

## How to Use This Document

The tutorial teaches **why** TurboQuant works. This document teaches **where** it is implemented. Every section below is tagged with the tutorial section it corresponds to, and every major design decision is explained in terms of the theory it realizes.

**Reading order recommendation:**
1. Tutorial Sections 1-3 (understand the problem)
2. This document, Chapters 1-2 (lloyd_max.py)
3. Tutorial Sections 4-5 (understand Stage 1 and Stage 2 theory)
4. This document, Chapter 3 (turboquant.py)
5. Tutorial Section 6 (understand the full estimator)
6. This document, Chapter 4 (compressors.py — production path)
7. Tutorial Section 7 (empirical results)
8. This document, Chapters 5-6 (test + validate scripts)

---

## Codebase Map

```
turboquant/
├── lloyd_max.py          ← The mathematical foundation: optimal codebook design
│                            (Tutorial §4: Random Rotation + Lloyd-Max)
├── turboquant.py         ← The algorithm core: Stage 1 + Stage 2 + KV cache wrapper
│                            (Tutorial §4, §5, §6: the full estimator)
├── compressors.py        ← Production path: handles real model tensors (B, H, S, D)
│                            (Tutorial §6: asymmetric attention without decompression)
├── test_turboquant.py    ← Synthetic validation: proves the math checks out
│                            (Tutorial §7: what the numbers mean)
└── validate.py           ← Real model validation: Qwen2.5-3B on RTX 3060
                             (Tutorial §7: empirical results)
```

**Two layers of implementation:**
- `turboquant.py` is the **pedagogical layer** — clean, modular, works on flat vectors `(batch, d)`. Good for understanding the algorithm.
- `compressors.py` is the **production layer** — handles real 4D model tensors `(B, H, S, D)`, pre-computes reconstructions for speed. Good for real use.

---

## Chapter 1: `lloyd_max.py` — The Optimal Codebook

*Tutorial connection: Section 4 ("Stage 1: Random Rotation + Lloyd-Max")*

This file answers one question: **given that rotated key vectors have coordinates that follow a known distribution, what is the best possible set of representative values (centroids) for quantization?**

### 1.1 The Coordinate Distribution

```python
# lloyd_max.py, lines 18-23
def beta_pdf(x: float, d: int) -> float:
    """PDF of a single coordinate after random rotation of a d-dim unit vector."""
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1 - x * x) ** ((d - 3) / 2)
```

**What this is:** After rotating a unit vector `x ∈ R^d` by a random orthogonal matrix `Π`, each coordinate of `Πx` follows a Beta distribution on `[-1, 1]`. This is the **exact** distribution — derived from the geometry of the hypersphere.

**Why it matters (Tutorial §4):** This is the foundation of why random rotation enables near-optimal scalar quantization. Because the distribution is known and fixed (it doesn't depend on the data), you can solve for the optimal codebook offline, once, and reuse it forever.

```python
# lloyd_max.py, lines 26-29
def gaussian_approx_pdf(x: float, d: int) -> float:
    """Gaussian approximation N(0, 1/d) -- accurate for d >= 64."""
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))
```

**The approximation in practice:** For the head dimensions typical in LLMs (`d = 64, 128`), the Beta distribution is nearly indistinguishable from `N(0, 1/d)`. The Gaussian approximation is used in `LloydMaxCodebook` by default (`use_exact=False`) — it's faster and accurate enough.

**σ = 1/√d is a key constant** — it appears throughout the codebase (in `solve_lloyd_max`, `_solve_codebook` in compressors.py). Every time you see `sigma = 1.0 / math.sqrt(d)`, the code is anchoring to this distribution.

---

### 1.2 The Lloyd-Max Algorithm

```python
# lloyd_max.py, lines 32-86 (condensed)
def solve_lloyd_max(d: int, bits: int, use_exact: bool = False, max_iter: int = 200, tol: float = 1e-10):
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)

    # Initialize centroids uniformly
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for iteration in range(max_iter):
        # STEP 1: Boundaries = midpoints between adjacent centroids
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

        # STEP 2: Centroids = conditional expectations E[X | X in partition_i]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _   = integrate.quad(lambda x: x * pdf(x), a, b)   # E[X·p(X)]
            denominator, _ = integrate.quad(pdf, a, b)                     # ∫p(X)dx
            new_centroids.append(numerator / denominator)

        if max(abs(new_centroids[i] - centroids[i]) ...) < tol:
            break
        centroids = new_centroids
```

**Mapping to Tutorial §4 — the two Lloyd-Max optimality conditions:**

| Condition | Mathematical Form | Code |
|---|---|---|
| **Boundaries = midpoints** | `b_i = (c_i + c_{i+1}) / 2` | `(centroids[i] + centroids[i+1]) / 2.0` (line 57) |
| **Centroids = conditional means** | `c_i = E[X \| b_{i-1} < X ≤ b_i]` | `numerator / denominator` (line 69) — numerator = `∫x·p(x)dx`, denominator = `∫p(x)dx` |

These two conditions are the definition of optimality for a scalar quantizer. The algorithm alternates between them until convergence — this is exactly continuous 1-D k-means.

**Initialization strategy:** Centroids start uniformly spread over `[-3.5σ, 3.5σ]`. This covers 3.5 standard deviations, which captures >99.9% of the probability mass of `N(0, 1/d)`. The `lo * 3` and `hi * 3` extended edges in the loop ensure numerical integration doesn't get truncated.

**Convergence:** `tol=1e-10` means the loop runs until centroids shift by less than 0.1 nanounits. In practice, it converges in 20-50 iterations for typical `(d, bits)` combinations.

---

### 1.3 The Symmetry Property

```python
# test_turboquant.py, lines 32-35
cb = LloydMaxCodebook(128, 3)
centroid_sum = cb.centroids.sum().abs().item()
print(f"Symmetry check: sum of centroids = {centroid_sum:.6f} (should be ~0)")
assert centroid_sum < 0.01, "Centroids should be symmetric!"
```

**Why this matters:** Because `N(0, 1/d)` is symmetric around zero, the optimal codebook is also symmetric: if `c` is a centroid, so is `-c`. This means the quantization error has zero mean — no systematic positive or negative bias from Stage 1 alone. (Note: *per-vector* inner product bias is different from *centroid* symmetry — Stage 1 still has biased inner products, which is why QJL is needed.)

---

### 1.4 `LloydMaxCodebook` — The Reusable Interface

```python
# lloyd_max.py, lines 107-131
class LloydMaxCodebook:
    def __init__(self, d: int, bits: int, use_exact: bool = False):
        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)
        self.distortion = compute_expected_distortion(...)  # theoretical MSE

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        diffs = (x.unsqueeze(-1) - self.centroids.to(x.device))  # (..., n_levels)
        return diffs.abs().argmin(dim=-1)                          # nearest centroid index

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(indices.device)[indices]
```

**Design note:** The codebook is computed once at construction time (expensive — involves numerical integration). After that, `quantize` and `dequantize` are pure tensor operations. `quantize` is a nearest-neighbor lookup: compute distance to all `2^bits` centroids, take argmin. This is `O(n_levels)` per coordinate — acceptable since `n_levels ≤ 16` for 4-bit.

**The `distortion` attribute** stores the theoretical expected MSE per coordinate. This is what `test_turboquant.py` compares against when validating that empirical MSE stays within the theoretical bound `D_mse ≤ √3·π/2 · (1/4^b)`.

---

## Chapter 2: `turboquant.py` — The Algorithm Core

*Tutorial connection: Sections 4, 5, and 6*

This file contains the complete TurboQuant algorithm in four components:

| Component | Tutorial | What it does |
|---|---|---|
| `generate_rotation_matrix()` | §4 | Produces the Haar-distributed orthogonal matrix Π |
| `generate_qjl_matrix()` | §5 | Produces the random Gaussian projection matrix S |
| `TurboQuantMSE` | §4 | Stage 1: rotate + quantize |
| `TurboQuantProd` | §5, §6 | Stage 1 + Stage 2: full unbiased estimator |
| `TurboQuantKVCache` | §6 | End-to-end KV cache with compressed attention |

---

### 2.1 `generate_rotation_matrix()` — The Haar Distribution

```python
# turboquant.py, lines 18-33
def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    G = torch.randn(d, d, generator=gen)       # Step 1: fill d×d matrix with N(0,1)
    Q, R = torch.linalg.qr(G)                  # Step 2: QR decomposition → Q is orthogonal

    # Step 3: Fix sign ambiguity — ensures det(Q) = +1 (a proper rotation)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)             # flip columns where diag(R) < 0

    return Q.to(device)
```

**Mapping to Tutorial §4 — the Haar distribution:**

The Haar measure on the orthogonal group O(d) is the unique distribution such that if `Q` is Haar-distributed, then `QA` and `AQ` have the same distribution as `Q` for any fixed orthogonal `A`. This is the "uniform distribution" over rotations.

**Why QR of a Gaussian matrix gives Haar:** If `G` has i.i.d. `N(0,1)` entries and `G = QR`, then `Q` is Haar-distributed on O(d). This is a classical result — Gaussian matrices are "rotation-invariant", so their QR factors inherit that invariance.

**The sign correction (line 30-32):** QR decomposition has sign ambiguity — `Q` and `-Q` both satisfy `G = QR`. Without correction, `Q` might have `det(Q) = -1` (a reflection, not a rotation). The fix: multiply each column of `Q` by `sign(R[i,i])` to make `R` have positive diagonal, ensuring `det(Q) = +1`. This is why `diag_sign[diag_sign == 0] = 1.0` handles the edge case of a zero diagonal entry.

**Reproducibility:** The `seed` parameter means the same rotation matrix is always generated for a given `(d, seed)` pair. This is critical — the query and key must use the **same** rotation matrix for the inner product to be meaningful. In `TurboQuantMSE`, the matrix is stored as a `register_buffer` (not a parameter — it doesn't need gradients).

---

### 2.2 `generate_qjl_matrix()` — The JL Projection

```python
# turboquant.py, lines 36-48
def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None,
                         device: str = "cpu") -> torch.Tensor:
    if m is None:
        m = d           # default: project to same dimensionality
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)   # S ∈ R^{m×d}, entries ~ N(0,1)
    return S.to(device)
```

**Mapping to Tutorial §5 — the Johnson-Lindenstrauss matrix:**

`S` is the random projection matrix. Its properties:
- Each row is an independent random direction in `R^d`
- `E[(S_i · x)(S_i · y)] = x · y` for any vectors `x, y` (unbiasedness of a single row)
- Summing `m` such estimates and dividing by `m` gives the JL inner product estimator

**Why `m = d` by default:** Using as many projection rows as the vector dimension gives the best variance/storage tradeoff for the QJL correction. The variance of the estimator scales as `O(1/m)`, so larger `m` is better — but uses more memory. Setting `m = d` means the QJL matrix is square, and the sign vector has the same length as the original vector: 1 bit per coordinate = d bits total.

**Different seed from rotation matrix:** In `TurboQuantProd`, the MSE quantizer uses `seed` and the QJL matrix uses `seed + 1`. This ensures statistical independence between Π and S — they must not be correlated for the unbiasedness proof to hold.

---

### 2.3 `TurboQuantMSE` — Stage 1

```python
# turboquant.py, lines 51-100
class TurboQuantMSE(nn.Module):
    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        # ① Store rotation matrix as a buffer (moved with .to(device), saved with state_dict)
        self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))

        # ② Precompute and store Lloyd-Max codebook
        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))
```

**Design note on `register_buffer`:** Both `Pi` and `centroids` are stored as buffers rather than parameters. This means:
- They move to the correct device when you call `.to(device)` or `.cuda()`
- They are saved and loaded with `state_dict()` / `load_state_dict()`
- They do NOT receive gradients — correct, since TurboQuant is a non-learned quantizer

```python
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T    # y = x Π^T  (note: Pi is stored as Π, so Pi.T = Π^T)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.Pi      # x = y Π   (because Π^T Π = I → inverse = Π)
```

**The rotation direction:** The code applies `x @ Pi.T` which computes `x Π^T`. Since `Π` is orthogonal, `Π^T = Π^{-1}`, so `unrotate` applies `y @ Pi = y Π` which recovers `x`. This is consistent: rotate with `Π^T`, undo with `Π`.

```python
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rotate(x)                              # y = x Π^T
        diffs = y.unsqueeze(-1) - self.centroids        # (..., d, n_levels)
        indices = diffs.abs().argmin(dim=-1)            # nearest centroid per coordinate
        return indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        y_hat = self.centroids[indices]                 # look up centroid values
        return self.unrotate(y_hat)                     # y_hat Π → back to original space
```

**The quantization as nearest-neighbor lookup:** `y.unsqueeze(-1) - self.centroids` broadcasts to shape `(..., d, n_levels)` — for each of the `d` coordinates, it computes the distance to all `n_levels` centroids simultaneously. `argmin` over the last dimension gives the index of the nearest centroid. This is a fully vectorized, batched nearest-neighbor search.

**The round-trip (forward):**
```python
    def forward(self, x):
        indices = self.quantize(x)      # x → rotated space → nearest centroid index
        x_hat = self.dequantize(indices)# index → centroid → unrotated space
        return x_hat, indices           # x_hat ≈ x, indices = compressed representation
```

**What is actually stored:** `indices` is a tensor of integers in `[0, 2^bits - 1]`. For `d=128`, `bits=3`, this is `128` integers each needing 3 bits = 384 bits = 48 bytes per vector. Compare to `128 × 2 = 256 bytes` for fp16 — about 5x smaller.

---

### 2.4 `TurboQuantProd` — Stage 1 + Stage 2 (The Full Algorithm)

*Tutorial connection: Sections 5 and 6 — this is the centerpiece of the implementation*

```python
# turboquant.py, lines 103-196
class TurboQuantProd(nn.Module):
    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None,
                 seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.mse_bits = max(bits - 1, 1)   # Stage 1 uses (bits - 1) bits
        self.qjl_dim = qjl_dim or d        # Stage 2 uses d bits (1 per projected dim)

        # Stage 1: MSE quantizer with (bits-1) bits
        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)

        # Stage 2: QJL projection matrix (different seed → independent of Π)
        self.register_buffer("S", generate_qjl_matrix(d, m=self.qjl_dim,
                                                        seed=seed + 1, device=device))
```

**The bit budget split (Tutorial §5):**

At a total budget of `b` bits per coordinate:
- Stage 1 (MSE): `(b-1)` bits → `2^(b-1)` Lloyd-Max levels
- Stage 2 (QJL): `1` bit → sign of the projected residual

For `bits=3`: Stage 1 uses 2-bit codebook (4 levels), Stage 2 uses 1 bit. Total = 3 bits/coordinate.
For `bits=4`: Stage 1 uses 3-bit codebook (8 levels), Stage 2 uses 1 bit. Total = 4 bits/coordinate.

The `max(bits - 1, 1)` ensures that at `bits=1`, you still get a 1-bit MSE quantizer (2 levels: just the sign). At `bits=2`, you get 1-bit MSE + 1-bit QJL.

---

#### 2.4.1 `quantize()` — What Gets Stored

```python
    def quantize(self, x: torch.Tensor) -> dict:
        # ── STAGE 1 ──────────────────────────────────────────────────────────
        x_hat, mse_indices = self.mse(x)          # rotate → quantize → dequantize
        #                                           x_hat = k_mse (the MSE reconstruction)

        residual = x - x_hat                       # r = k - k_mse
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)  # ‖r‖

        # ── STAGE 2 ──────────────────────────────────────────────────────────
        projected = residual @ self.S.T            # Sr  (shape: batch × qjl_dim)
        qjl_signs = torch.sign(projected)          # sign(Sr)
        qjl_signs[qjl_signs == 0] = 1.0           # map rare 0 → +1

        return {
            "mse_indices":   mse_indices,          # (batch, d)   — (b-1) bits per element
            "qjl_signs":     qjl_signs,            # (batch, d)   — 1 bit per element
            "residual_norm": residual_norm.squeeze(-1),  # (batch,) — 16 bits (fp16 in V2)
        }
```

**Line by line:**

1. `x_hat, mse_indices = self.mse(x)` — runs the full Stage 1 round-trip. `x_hat` is the MSE reconstruction (what you'd get if you only used Stage 1), `mse_indices` are the compressed integers to store.

2. `residual = x - x_hat` — the quantization error from Stage 1. In the tutorial notation, this is `r_k = k - k_mse`. The key insight: *this residual still carries information about the true inner product* — and that's exactly what Stage 2 captures.

3. `residual_norm = torch.norm(residual, dim=-1, keepdim=True)` — the L2 norm of the residual, `‖r_k‖`. This is stored in 16 bits (fp16 in the V2 compressor). It tells Stage 2 how large the correction needs to be — a tiny residual needs almost no correction; a large residual needs more.

4. `projected = residual @ self.S.T` — project the residual through the JL matrix. Shape: `(batch, qjl_dim)`. Each element is one inner product `S_i · r`.

5. `qjl_signs = torch.sign(projected)` — discard everything except the sign. This is the 1-bit compression. Each element is now `+1` or `-1`.

6. `qjl_signs[qjl_signs == 0] = 1.0` — edge case: `torch.sign(0) = 0`, but we need `±1`. Map to `+1` arbitrarily (probability zero for continuous distributions, but floating point can produce exact zeros).

**What is NOT stored:** The actual residual vector `r` (it's d-dimensional in fp16 = 256 bytes — we discard it). The MSE reconstruction `x_hat` (also discarded — can be recomputed from `mse_indices` during inner product estimation). Only the three compressed components are kept.

---

#### 2.4.2 `inner_product()` — The Asymmetric Estimator

*This is the mathematical heart of TurboQuant. Every line maps directly to a term in the tutorial's central formula.*

```python
    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Estimate ⟨y, x⟩ using compressed x.

        The formula (Tutorial §6):
          ⟨q, k⟩ ≈ ⟨q, k_mse⟩  +  ‖r_k‖ · √(π/2)/m · ⟨Sq, sign(Sr_k)⟩
                    ─ Term 1 ─     ──────────────── Term 2 ──────────────────
        """
        # ── TERM 1 ───────────────────────────────────────────────────────────
        x_mse = self.mse.dequantize(compressed["mse_indices"])  # recover k_mse
        term1 = (y * x_mse).sum(dim=-1)                         # ⟨q, k_mse⟩

        # ── TERM 2 ───────────────────────────────────────────────────────────
        y_projected = y @ self.S.T                               # S·q  (project query)
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)  # ⟨Sq, sign(Sr_k)⟩

        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m           # √(π/2) / m
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip

        return term1 + term2
```

**Annotated mapping to Tutorial §6:**

| Code | Formula | What it represents |
|---|---|---|
| `x_mse = self.mse.dequantize(...)` | `k_mse` | MSE reconstruction from Stage 1 |
| `(y * x_mse).sum(dim=-1)` | `⟨q, k_mse⟩` | Inner product with the compressed part |
| `y @ self.S.T` | `Sq` | Project the full-precision query through S |
| `(y_projected * compressed["qjl_signs"]).sum(-1)` | `⟨Sq, sign(Sr_k)⟩` | Dot product between projected query and stored signs |
| `math.sqrt(math.pi / 2) / m` | `√(π/2)/m` | The JL correction factor |
| `compressed["residual_norm"] * correction_scale * qjl_ip` | `‖r_k‖ · √(π/2)/m · ⟨Sq, sign(Sr_k)⟩` | The full QJL correction term |
| `term1 + term2` | `⟨q, k⟩` | The unbiased estimate |

**Why `√(π/2)/m`? (Tutorial §5):**

For `z ~ N(0, 1)`, we have `E[|z|] = √(2/π)`. Therefore `E[sign(z) · w] = E[z · w] / E[|z|]` for any `w` independent of the sign choice — but this isn't quite right. The correct derivation:

For `a = S_i · r` (a scalar) and a query projection `b = S_i · q`:
```
E[sign(a) · b] = E[sign(a)] · E[b | a>0] - E[sign(a)] · E[b | a<0]
               = (2/√(2π)) · (a/|a|) · ...
```

More directly: for the estimator `(π/2)/m · ∑_i sign(S_i·r) · (S_i·q)`, the expectation over the random matrix S is exactly `⟨q, r⟩`. The factor `√(π/2)/m` is the correction that makes `E[sign(a)·b] = ⟨q, r⟩/m · ...` work out to be unbiased. It is the reciprocal of `E[|z|/σ]` for `z ~ N(0,σ²)`.

**"Asymmetric" explained:** The query `y` is used at **full precision** — it is projected through `S` without quantization. Only the key `x` is compressed (its MSE reconstruction and QJL signs are stored). This asymmetry is why the estimator works: the query contributes exact information; the key contributes compressed information. If both were quantized, the error would compound.

---

### 2.5 `TurboQuantKVCache` — The End-to-End Wrapper

```python
# turboquant.py, lines 199-286
class TurboQuantKVCache:
    def __init__(self, d_key, d_value, bits=3, seed=42, device="cpu"):
        # Keys use TurboQuantProd (inner products needed for attention scores)
        self.key_quantizer   = TurboQuantProd(d_key,   bits, seed=seed,       device=device)
        # Values use TurboQuantMSE (MSE reconstruction needed for weighted sum)
        self.value_quantizer = TurboQuantMSE(d_value,  bits, seed=seed + 100, device=device)
```

**The keys vs values asymmetry (Tutorial §6):**

This is a fundamental design decision that the tutorial explains but the code makes concrete:

- **Keys** need `TurboQuantProd` (Stage 1 + QJL) because attention score computation is `softmax(Q @ K^T / √d)` — inner products between queries and keys. You need unbiased inner product estimates, which requires the QJL correction.

- **Values** need only `TurboQuantMSE` (Stage 1 only) because value aggregation is `softmax_weights @ V` — a weighted sum of value vectors. The softmax weights are computed from attention scores (which are separately corrected by QJL). The weighted sum of values averages out per-vector reconstruction errors, making the MSE-only approach sufficient.

Different seeds (`seed` vs `seed + 100`) ensure the rotation matrices for keys and values are independent.

```python
    def memory_usage_bits(self) -> dict:
        n_keys = sum(c["mse_indices"].numel() for c in self.key_cache)
        n_qjl  = sum(c["qjl_signs"].numel()   for c in self.key_cache)
        n_norms = sum(c["residual_norm"].numel() for c in self.key_cache)
        n_values = sum(c["indices"].numel() for c in self.value_cache)

        # Keys:   (b-1) bits per MSE index + 1 bit per QJL sign + 16 bits per residual norm
        key_bits   = n_keys * self.key_quantizer.mse_bits + n_qjl * 1 + n_norms * 16
        # Values: b bits per MSE index (no QJL)
        value_bits = n_values * self.bits
        # Baseline: fp16 for everything
        fp16_equivalent = (n_keys + n_values) * 16
```

**The bit accounting in full detail (Tutorial §7):**

For `b=3` bits, `d=128`, `N` total key/value vectors:

| Component | Size | Bits |
|---|---|---|
| MSE indices (keys) | `N × d` integers in `[0, 3]` | `N × d × 2` = `N × 256` bits |
| QJL signs (keys) | `N × d` values of `±1` | `N × d × 1` = `N × 128` bits |
| Residual norms (keys) | `N` scalars in fp16 | `N × 16` bits |
| MSE indices (values) | `N × d` integers in `[0, 7]` | `N × d × 3` = `N × 384` bits |
| **Total** | | `N × (256 + 128 + 16 + 384) = N × 784` bits |
| **fp16 baseline** | `2N × d` values × 16 bits | `N × 2 × 128 × 16 = N × 4096` bits |
| **Compression ratio** | `4096 / 784` | **~5.2x** |

This matches the transcript's "5.22x at 3-bit" result exactly.

---

## Chapter 3: `compressors.py` — The Production Path

*Tutorial connection: Section 6 ("The Full Estimator — Asymmetric Attention Without Decompression")*

While `turboquant.py` works on flat `(batch, d)` vectors, `compressors.py` is designed for real model tensors: `(B, H, S, D)` = (batch, heads, sequence, head_dim). It also makes one important optimization: **pre-computing `k_mse` during compression** rather than recomputing it during every inner product query.

---

### 3.1 `TurboQuantCompressorV2` — The Key Compressor

#### Construction: Three Random Objects

```python
# compressors.py, lines 31-55
class TurboQuantCompressorV2:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.mse_bits = max(bits - 1, 1)

        # ① Rotation matrix Π (same construction as generate_rotation_matrix)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)

        # ② Lloyd-Max codebook (inlined version of solve_lloyd_max)
        self.centroids = self._solve_codebook(head_dim, self.mse_bits).to(device)

        # ③ QJL matrix S (square: head_dim × head_dim)
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)          # note: +10000 not +1 as in turboquant.py
        self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)

        # ④ Precomputed transpose for speed
        self.PiT = self.Pi.T.contiguous()
```

**Difference from `turboquant.py`:** The seed offset for S is `+10000` here vs `+1` in `TurboQuantProd`. Both achieve the same goal (independence between Π and S), just with different offsets.

**`self.PiT = self.Pi.T.contiguous()`:** Pre-computes and stores `Π^T` as a contiguous tensor. The `.contiguous()` call ensures the memory layout is optimal for matrix multiplication — transposed tensors in PyTorch are not always stored contiguously, which can slow down matmul.

---

#### `compress()` — The Full Pipeline in One Pass

```python
# compressors.py, lines 83-120
@torch.no_grad()
def compress(self, states: torch.Tensor) -> dict:
    B, H, S, D = states.shape
    flat = states.reshape(-1, D).float()      # flatten batch/head/seq → (N, D)

    # ── NORMALIZE ────────────────────────────────────────────────────────────
    vec_norms = torch.norm(flat, dim=-1, keepdim=True)   # ‖k‖ for each vector
    flat_norm = flat / (vec_norms + 1e-8)                # k̂ = k / ‖k‖  (unit vectors)

    # ── STAGE 1: ROTATE + QUANTIZE ───────────────────────────────────────────
    rotated = flat_norm @ self.Pi.T                      # ŷ = k̂ Π^T
    diffs = rotated.unsqueeze(-1) - self.centroids       # distance to all centroids
    indices = diffs.abs().argmin(dim=-1).to(torch.uint8) # nearest centroid index

    # ── STAGE 1: MSE RECONSTRUCTION ──────────────────────────────────────────
    reconstructed_rotated = self.centroids[indices.long()]   # ŷ_hat (in rotated space)
    k_mse = (reconstructed_rotated @ self.Pi) * vec_norms    # k_mse = ŷ_hat Π · ‖k‖

    # ── STAGE 2: RESIDUAL + QJL ──────────────────────────────────────────────
    residual = flat - k_mse                              # r = k - k_mse
    residual_norm = torch.norm(residual, dim=-1)         # ‖r‖

    projected = residual @ self.S.T                      # S · r
    signs = (projected >= 0).to(torch.int8) * 2 - 1    # sign(Sr): +1 or -1 as int8

    return {
        "k_mse":        k_mse.to(torch.float16).reshape(B, H, S, D),
        "qjl_signs":    signs.reshape(B, H, S, D),
        "residual_norm": residual_norm.to(torch.float16).reshape(B, H, S),
        "shape":        (B, H, S, D),
    }
```

**Key difference from `TurboQuantProd.quantize()`:** This version **pre-computes `k_mse`** and stores it in fp16 instead of storing `mse_indices`. This trades memory for speed — during attention score computation, you don't need to do a codebook lookup; you just use the already-reconstructed `k_mse` directly.

**The normalization step (not in `turboquant.py`):** `flat_norm = flat / (vec_norms + 1e-8)` normalizes each vector to unit norm before rotation and quantization. The original norm is stored as `vec_norms` (in fp16, one scalar per vector). During reconstruction, the norm is multiplied back: `k_mse = ... * vec_norms`.

Why normalize? The Lloyd-Max codebook was designed for unit vectors (distribution N(0, 1/d) after rotating a unit vector). Real key vectors from LLMs are not unit vectors — they have varying magnitudes. Normalizing before quantization and restoring the norm after brings real vectors into the regime where the codebook is optimal.

**`signs = (projected >= 0).to(torch.int8) * 2 - 1`:** This is the int8-efficient way to compute `sign(projected)`. `(projected >= 0)` gives `True/False` (1/0), multiplying by 2 gives `2/0`, subtracting 1 gives `1/-1`. Stored as `int8` rather than float32 — 4x smaller, and the sign operation is still exactly correct.

---

#### `asymmetric_attention_scores()` — Batched Attention Over Compressed Keys

```python
# compressors.py, lines 122-158
@torch.no_grad()
def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
    k_mse  = compressed["k_mse"].float()            # (B, H, S_k, D)
    signs  = compressed["qjl_signs"].float()         # (B, H, S_k, D)
    r_norm = compressed["residual_norm"].float()     # (B, H, S_k)

    # ── TERM 1: Q @ K_mse^T ──────────────────────────────────────────────────
    term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
    #       (B, H, S_q, D) @ (B, H, D, S_k) → (B, H, S_q, S_k)

    # ── TERM 2: QJL CORRECTION ───────────────────────────────────────────────
    q_projected = torch.matmul(queries.float(), self.S.T)
    #             (B, H, S_q, D) @ (D, D) → (B, H, S_q, D)

    qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))
    #         (B, H, S_q, D) @ (B, H, D, S_k) → (B, H, S_q, S_k)

    m = self.S.shape[0]                             # = head_dim
    correction_scale = math.sqrt(math.pi / 2) / m  # √(π/2) / m

    term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)
    #       r_norm: (B, H, S_k) → (B, H, 1, S_k) via unsqueeze(-2)
    #       Broadcasting: (B, H, S_q, S_k) * (B, H, 1, S_k) → (B, H, S_q, S_k)

    return term1 + term2                            # (B, H, S_q, S_k)
```

**The broadcasting trick:** `r_norm.unsqueeze(-2)` reshapes `(B, H, S_k)` to `(B, H, 1, S_k)`. When multiplied against `qjl_ip` of shape `(B, H, S_q, S_k)`, broadcasting repeats the norms across all query positions. This is correct — the residual norm belongs to each key, not each query.

**All three matrix multiplications are batched:** PyTorch's `torch.matmul` on 4D tensors applies the matrix multiplication independently for each `(B, H)` slice. This means all heads and all batch elements are processed in parallel on the GPU.

**Why `@torch.no_grad()`:** TurboQuant is inference-only — no gradients needed. The decorator avoids building a computation graph, saving memory and time.

---

### 3.2 `TurboQuantCompressorMSE` — The Value Compressor

```python
# compressors.py, lines 161-221
class TurboQuantCompressorMSE:
    # Same rotation matrix + codebook setup as V2, but no QJL matrix

    def compress(self, states):
        # Same normalization + rotation + quantization as V2
        # But stores indices + vec_norms instead of k_mse + signs + residual_norm
        return {
            "indices":   indices,               # (N,) uint8
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "shape":     (B, H, S, D),
        }

    def decompress(self, compressed):
        indices = compressed["indices"].long()
        reconstructed = self.centroids[indices] @ self.Pi      # look up centroids, unrotate
        vec_norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (reconstructed * vec_norms).reshape(B, H, S, D)  # restore scale
```

**Comparison with V2:**

| Aspect | `TurboQuantCompressorV2` (keys) | `TurboQuantCompressorMSE` (values) |
|---|---|---|
| Stores | `k_mse` + `signs` + `residual_norm` | `indices` + `vec_norms` |
| Has QJL | Yes | No |
| Usage | `asymmetric_attention_scores()` | `decompress()` |
| Why | Need unbiased inner products | Need vector reconstruction |

Note that V2 stores `k_mse` (the pre-reconstructed fp16 vector), while MSE stores raw `indices`. V2 trades more storage for faster attention score computation (no lookup at query time). MSE stores indices to save memory and does the lookup only at decompression time.

---

## Chapter 4: `test_turboquant.py` — Validating the Math

*Tutorial connection: Section 7 ("Empirical Results") — these are the "synthetic validation" results*

The test script has 7 tests, each validating a specific mathematical claim. Here they are mapped to the underlying theory:

---

### Test 1: Lloyd-Max Codebook Properties

```python
# test_turboquant.py, lines 18-36
def test_lloyd_max_codebook():
    for d in [64, 128, 256]:
        for bits in [1, 2, 3, 4]:
            cb = LloydMaxCodebook(d, bits)
            # checks: n_levels, distortion/coord, centroids range

    # Key assertion: codebook is symmetric around zero
    centroid_sum = cb.centroids.sum().abs().item()
    assert centroid_sum < 0.01
```

**What it proves:** For each `(d, bits)` combination, the codebook was computed correctly, and it is symmetric around zero (sum of all centroids ≈ 0). The symmetry check catches bugs where the distribution was mis-specified or the iteration didn't converge.

---

### Test 2: MSE Distortion Bounds

```python
# test_turboquant.py, lines 39-71
def test_mse_quantizer():
    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(d=128, bits=bits)
        x = torch.randn(1000, 128); x /= x.norm(dim=-1, keepdim=True)
        x_hat, _ = quantizer(x)
        mse = ((x - x_hat)**2).sum(dim=-1).mean()

        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / 4**bits)
        ratio = mse / theoretical_bound   # should be < 1.0
```

**What it proves:** The empirical MSE (measured on 1000 random unit vectors) is below the paper's theoretical upper bound `D_mse ≤ √3·π/2 · (1/4^b)`. The `ratio < 1.0` confirms Stage 1's optimality. From the README, the measured ratios are 0.53x, 0.68x, 0.81x, 0.87x at 1-4 bits respectively — all well under the bound.

---

### Test 3: Inner Product Unbiasedness

```python
# test_turboquant.py, lines 74-113
def test_inner_product_unbiasedness():
    for bits in [2, 3, 4]:
        quantizer = TurboQuantProd(d=128, bits=bits)
        # ... generate x, y ...
        true_ip     = (x * y).sum(dim=-1)
        estimated_ip = quantizer.inner_product(y, quantizer.quantize(x))

        bias        = (estimated_ip - true_ip).mean()       # should be ≈ 0
        correlation = torch.corrcoef(stack([true_ip, estimated_ip]))[0,1]
```

**What it proves:** The TurboQuantProd estimator is unbiased — the average error across 2000 vector pairs is near zero. The correlation measures how well the estimates track the true values (0.80 at 2-bit, 0.93 at 3-bit, 0.98 at 4-bit from the README).

---

### Test 4: MSE-Only Bias (The Motivation for QJL)

```python
# test_turboquant.py, lines 116-143
def test_mse_only_inner_product_bias():
    for bits in [1, 2, 3]:
        quantizer = TurboQuantMSE(d=128, bits=bits)
        x_hat, _ = quantizer(x)
        mse_ip = (x_hat * y).sum(dim=-1)
        bias = (mse_ip - true_ip).mean()   # NOT near zero — MSE-only is biased!
```

**What it proves:** Without QJL (Stage 2), the inner products computed from MSE-reconstructed vectors are **systematically biased**. The bias is negative (MSE compression shrinks the magnitudes of vectors, so inner products are underestimated). This is the fundamental motivation for adding Stage 2 — this test makes the problem visible.

This test directly supports Tutorial §5's explanation of why two stages are necessary.

---

### Test 5: KV Cache Compression Ratios

```python
# test_turboquant.py, lines 147-177
def test_kv_cache():
    for bits in [2, 3, 4]:
        cache = TurboQuantKVCache(d_key=128, d_value=128, bits=bits)
        cache.append(keys, values)          # seq_len=1024 vectors
        usage = cache.memory_usage_bits()   # measure actual bits used
        print(f"bits={bits}: compression={usage['compression_ratio']:.2f}x")
```

**What it proves:** The compression ratios match the theoretical analysis — ~7.76x at 2-bit, ~5.22x at 3-bit, ~3.94x at 4-bit. It also tests the full `attention_scores()` pipeline end-to-end.

---

### Test 6: Needle-in-Haystack

```python
# test_turboquant.py, lines 181-223
def test_needle_in_haystack():
    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            needle_pos = seq_len // 3
            query = keys[needle_pos].clone().unsqueeze(0)  # exact copy of one key

            estimated_ips = quantizer.inner_product(
                query.expand(seq_len, -1), quantizer.quantize(keys)
            )
            top_idx = estimated_ips.argmax()
            found = (top_idx == needle_pos)  # should be True every time
```

**What it proves:** Even after quantization, the inner product estimator correctly identifies the most similar vector in a sequence of 8,192 vectors. This is the "retrieval accuracy" test — confirming that TurboQuant preserves the ranking structure of inner products. Result: 9/9 exact matches across all configurations.

---

### Test 7: GPU Benchmark

```python
# test_turboquant.py, lines 227-288
def test_gpu_if_available():
    # Benchmark: quantization speed + inner product speed + fp16 matmul comparison
    # Reports: time per operation, memory comparison
```

This test benchmarks the Python implementation's speed on GPU vs full-precision matmul. It does NOT claim to match the paper's 8x speedup — this implementation is not CUDA-optimized. The speedup claim in the paper comes from a custom CUDA kernel, not a Python loop.

---

## Chapter 5: `validate.py` — Real Model Validation

*Tutorial connection: Section 7 ("Empirical Results — What 3-Bit Actually Buys You")*

This is the "does it actually work on a real model?" test. It loads Qwen2.5-3B-Instruct (in 4-bit weight quantization to fit in memory), runs a forward pass, captures the real KV cache, compresses it with TurboQuant, and compares attention scores.

### 5.1 The Needle-in-Haystack Prompt Setup

```python
# validate.py, lines 20-40
NEEDLE   = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"
FILLER   = """The quarterly financial review meeting covered several topics..."""

def build_prompt(tokenizer, target_tokens=2048, needle_pos=0.5):
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)         # insert needle at 50% depth by default
    # ... assembles haystack with needle embedded in the middle ...
```

**Why this prompt:** The needle-in-haystack test checks if attention is preserved for a specific, isolated fact buried in irrelevant text. If TurboQuant scrambles attention patterns, the model would fail to find "AURORA-7749". The needle is inserted at `needle_pos=0.5` (middle of the document) — a hard position that requires attending far back in the context.

---

### 5.2 Capturing Real KV Cache

```python
# validate.py, lines 83-90
with torch.no_grad():
    outputs = model(**inputs, use_cache=True, output_attentions=False)
cache = outputs.past_key_values

n_layers = len(cache.layers)               # 36 for Qwen2.5-3B
head_dim = cache.layers[0].keys.shape[-1]  # 128
num_kv_heads = cache.layers[0].keys.shape[1]  # 2 (GQA: grouped-query attention)
```

The model is run in inference mode (`torch.no_grad()`), with `use_cache=True` to populate the KV cache. `outputs.past_key_values` is the HuggingFace DynamicCache object — iterating over `cache.layers` gives per-layer `(keys, values)` tensors of shape `(1, num_kv_heads, seq_len, head_dim)`.

---

### 5.3 The Per-Layer Compression Loop

```python
# validate.py, lines 107-172 (condensed)
for layer_idx in range(n_layers):
    keys   = cache.layers[layer_idx].keys    # (1, H, S, D)
    values = cache.layers[layer_idx].values

    # Fresh compressor per layer — different seed for each
    key_comp = TurboQuantCompressorV2(D, bits, seed=layer_idx * 1000, device="cuda")
    val_comp = TurboQuantCompressorMSE(D, bits, seed=layer_idx * 1000 + 500, device="cuda")

    compressed_k = key_comp.compress(keys)
    compressed_v = val_comp.compress(values)
```

**Layer-specific compressors:** Each layer gets its own rotation matrix and codebook (via unique seeds `layer_idx * 1000`). In practice the same matrices could be shared across layers, but using per-layer matrices is more conservative and avoids any cross-layer correlation.

---

### 5.4 Simulating Next-Token Attention

```python
# validate.py, lines 138-144
# The query: last token attending to all previous tokens
query = keys[:, :, -1:, :]                 # (1, H, 1, D) — last token's key as proxy query

# Real attention scores
real_scores = torch.matmul(
    query.float(), keys.float().transpose(-2, -1)
).squeeze(-2)                               # (1, H, S)

# TurboQuant attention scores
tq_scores = key_comp.asymmetric_attention_scores(query, compressed_k).squeeze(-2)
```

**The "query as last key" approximation:** Ideally, you'd compare using the actual query projections from the model. Those require access to the attention module's weight matrices. Instead, the validation uses the last token's key vector as a stand-in query — this is a reasonable proxy since key and query projections come from the same hidden state and have similar statistics.

---

### 5.5 Metrics and What They Mean

```python
# validate.py, lines 147-179
for h in range(H):
    rs = real_scores[0, h]    # real attention logits for head h, shape (S,)
    ts = tq_scores[0, h]      # TQ-estimated attention logits

    # Metric 1: How similar are the full attention distributions?
    cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()

    # Metric 2: Does the most-attended token stay the same?
    real_top1 = rs.argmax(); tq_top1 = ts.argmax()
    if real_top1 == tq_top1: top1_matches += 1

    # Metric 3: Is the real top-1 still in the top 5 after compression?
    tq_top5 = ts.topk(5).indices.tolist()
    if real_top1 in tq_top5: top5_matches += 1

    # Metric 4: What rank does the needle get?
    needle_rank = (ts.argsort(descending=True) == needle_start).nonzero()
```

**What these metrics tell you:**

- **Cosine similarity 0.9945 at 3-bit** means the attention logit vectors (before softmax) are 99.45% directionally aligned. After softmax, the probability distributions will be even closer because softmax amplifies large values and suppresses small ones — if the ranking is mostly preserved, the distribution is very close.

- **Top-1 match 86% at 3-bit, 8K context** means that in 86 out of 100 layer-head combinations, the single most attended token is the same before and after compression. The 14% that differ likely attend to tokens with very similar scores — a small absolute change in logits can flip the argmax between close candidates.

- **Top-5 match 94%** means that even when the exact top-1 shifts, it almost always shifts to the second or third most attended token — not some random far-away position. Attention behavior is highly preserved.

---

## Chapter 6: Design Decisions and Their Theoretical Roots

A summary of non-obvious implementation choices, each tied back to the underlying theory:

### 6.1 Why QR Decomposition for the Rotation Matrix?

The QR decomposition of a Gaussian matrix gives the Haar distribution on O(d) — the unique rotation-invariant distribution. Any rotation matrix would "rotate" the vectors, but only a Haar-distributed one guarantees that the resulting coordinates have the exact distribution the Lloyd-Max codebook was designed for. Using a non-Haar distribution would make the codebook sub-optimal for the actual coordinate distribution.

### 6.2 Why `sign(projected) = ±1` and Not a Multi-Bit Code?

The 1-bit constraint is fundamental to the QJL unbiasedness result. The correction factor `√(π/2)/m` is derived specifically for the case where you store only the sign — it is `E[|z|]^{-1}` for `z ~ N(0,1)`, which is the bias correction needed for `sign(z)`. If you stored 2-bit codes, the correction factor and estimator formula would be different. The power of QJL is that **one bit per projected dimension is sufficient for unbiasedness** — you don't need more.

### 6.3 Why `(bits - 1)` for Stage 1 and 1 for Stage 2?

The total bit budget is `b` bits per coordinate. Stage 2 (QJL) is fixed at exactly 1 bit — it captures binary information (sign of residual projection). Giving Stage 2 more than 1 bit would require a fundamentally different estimator. So Stage 1 gets the remaining `(b-1)` bits. This is optimal: more MSE bits → smaller residual → smaller Stage 2 correction needed. The two stages are complementary: Stage 1 handles the signal, Stage 2 handles the bias.

### 6.4 Why Pre-compute `k_mse` in `compressors.py` but Not in `turboquant.py`?

`turboquant.py` stores `mse_indices` (compressed integers). `compressors.py` stores `k_mse` (fp16 reconstructed vectors). The tradeoff:

- `mse_indices`: smaller storage (`(b-1)` bits/coordinate), but requires a codebook lookup at query time
- `k_mse`: larger storage (16 bits/coordinate), but attention scores are computed with a single matmul

For the validation use case, `k_mse` is pre-computed once and reused for every query. This makes `asymmetric_attention_scores()` a pair of matmuls — fast on GPU, especially for batched heads.

### 6.5 Why Normalize Before Quantizing in `compressors.py`?

The Lloyd-Max codebook was designed for unit vectors (after rotation, coordinates follow N(0, 1/d)). Real LLM key vectors have varying L2 norms — they are not unit vectors. The normalize-then-quantize-then-scale-back approach brings every vector into the regime where the codebook is accurate, then restores the true scale at the end. Without normalization, large-magnitude vectors would have most coordinates outside the codebook's range, and small-magnitude vectors would all collapse to the center centroids — both are sub-optimal.

### 6.6 Why Do Values Use MSE-Only Compression?

The value aggregation in attention is: `output = softmax(scores) @ V`. This is a weighted sum of value vectors, where the weights are the softmax-normalized attention scores. If each value vector has an independent quantization error `ε_i`, and the weights `w_i` sum to 1, then the error in the output is `Σ w_i ε_i`. By the law of large numbers (over many tokens), this weighted sum averages out the per-vector errors — the output error is much smaller than any individual `ε_i`. This self-averaging property means you don't need QJL's unbiasedness for values; MSE-optimal reconstruction is enough.

---

## Quick Reference: Formula → Code Mapping

| Tutorial Formula | File | Function | Line |
|---|---|---|---|
| `N(0, 1/d)` coordinate distribution | `lloyd_max.py` | `gaussian_approx_pdf` | 26 |
| Lloyd-Max: `b_i = (c_i + c_{i+1})/2` | `lloyd_max.py` | `solve_lloyd_max` | 57 |
| Lloyd-Max: `c_i = E[X\|partition_i]` | `lloyd_max.py` | `solve_lloyd_max` | 65-69 |
| `Π` = Haar random rotation | `turboquant.py` | `generate_rotation_matrix` | 18-33 |
| `S` = i.i.d. N(0,1) JL matrix | `turboquant.py` | `generate_qjl_matrix` | 36-48 |
| `y = x Π^T` (rotate) | `turboquant.py` | `TurboQuantMSE.rotate` | 73 |
| `x̂ = ŷ Π` (unrotate) | `turboquant.py` | `TurboQuantMSE.unrotate` | 77 |
| `k_mse` = Stage 1 reconstruction | `turboquant.py` | `TurboQuantProd.quantize` | 144 |
| `r_k = k - k_mse` | `turboquant.py` | `TurboQuantProd.quantize` | 147 |
| `‖r_k‖` | `turboquant.py` | `TurboQuantProd.quantize` | 148 |
| `sign(Sr_k)` | `turboquant.py` | `TurboQuantProd.quantize` | 152-153 |
| `⟨q, k_mse⟩` (Term 1) | `turboquant.py` | `TurboQuantProd.inner_product` | 180-181 |
| `Sq` = projected query | `turboquant.py` | `TurboQuantProd.inner_product` | 185 |
| `⟨Sq, sign(Sr_k)⟩` | `turboquant.py` | `TurboQuantProd.inner_product` | 186 |
| `√(π/2)/m` correction factor | `turboquant.py` | `TurboQuantProd.inner_product` | 189 |
| `‖r_k‖ · √(π/2)/m · ⟨Sq, sign(Sr_k)⟩` (Term 2) | `turboquant.py` | `TurboQuantProd.inner_product` | 190 |
| `⟨q,k⟩ ≈ Term1 + Term2` | `turboquant.py` | `TurboQuantProd.inner_product` | 192 |
| Batched `Q @ K_mse^T` (Term 1) | `compressors.py` | `asymmetric_attention_scores` | 142 |
| Batched `Q @ S^T` (project queries) | `compressors.py` | `asymmetric_attention_scores` | 147 |
| Batched `⟨Sq, signs⟩` (Term 2 inner) | `compressors.py` | `asymmetric_attention_scores` | 150 |
| `r_norm` broadcast for Term 2 | `compressors.py` | `asymmetric_attention_scores` | 156 |

---

*This document is a companion to [`turboquant_tutorial.md`](turboquant_tutorial.md). Read both together for full theory-to-code coverage.*
