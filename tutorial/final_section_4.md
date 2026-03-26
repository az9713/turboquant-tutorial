# Section 4: Stage 1 -- The Random Rotation Trick + Lloyd-Max Optimal Scalar Quantization

We now arrive at the first of TurboQuant's two stages. The idea is elegant: if you could make every coordinate of your vector follow the same known distribution, you could design one fixed codebook that works optimally for all of them -- no per-block normalization, no scale factors, no overhead. Stage 1 achieves this with two operations: a random orthogonal rotation that makes coordinates approximately independent and identically distributed, followed by the Lloyd-Max algorithm that builds the optimal scalar quantizer for that known distribution.

## Why Random Rotation Makes Quantization Work

Imagine a 128-dimensional key vector whose coordinates have wildly different variances. Some coordinates carry large values, others are near zero, and there may be correlations between them. If you apply per-coordinate scalar quantization directly, the codebook would need to accommodate all these different scales -- which is exactly the per-block normalization problem from Section 3.

Now multiply that vector by a random orthogonal matrix Pi. Something remarkable happens: the rotated vector's coordinates become approximately **independent and identically distributed (i.i.d.)**. Each coordinate follows approximately N(0, 1/d) for a unit-norm vector in d dimensions.

Why does this work? The key property is that if x has a rotationally symmetric distribution (like N(0, sigma^2 * I)), then Pi @ x has exactly the same distribution for any orthogonal Pi. For unit-norm vectors, random rotation effectively "spreads" the information uniformly across all coordinates. No single coordinate dominates; no pair is correlated. The mathematical foundation is that a random rotation drawn from the **Haar measure** (the uniform distribution over orthogonal matrices) is the unique transformation that achieves this uniformity.

After rotation, a single fixed codebook optimized for N(0, 1/d) works for every coordinate. No normalization constants needed. This is why Stage 1 eliminates the overhead that plagues classical product quantization.

## Generating the Rotation Matrix

The `generate_rotation_matrix()` function in `turboquant.py` (lines 18-33) constructs a Haar-distributed random orthogonal matrix using QR decomposition:

```python
def generate_rotation_matrix(d, seed=None, device="cpu"):
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)       # Step 1: Random Gaussian matrix
    Q, R = torch.linalg.qr(G)                   # Step 2: QR decomposition
    diag_sign = torch.sign(torch.diag(R))        # Step 3: Sign correction
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)               # Ensure Haar measure
    return Q.to(device)
```

The procedure works in three steps:

1. **Generate a random Gaussian matrix G**: A d x d matrix with i.i.d. N(0,1) entries. This matrix is almost surely full rank.
2. **QR decomposition**: Factor G = Q @ R, where Q is orthogonal and R is upper triangular. The Q factor is "almost" Haar-distributed, but QR decomposition has a sign ambiguity -- the signs of the diagonal of R are arbitrary.
3. **Sign correction**: Multiply each column of Q by the sign of the corresponding diagonal element of R. This correction is mathematically necessary to ensure Q is truly uniformly distributed over the orthogonal group (Haar measure). Without it, the distribution would be biased.

The rotation matrix is computed once and stored as a PyTorch buffer (`register_buffer("Pi", ...)`), meaning it persists with the model but is not a learnable parameter. The same matrix is used for both rotation (`x @ Pi^T`) and un-rotation (`y @ Pi`), since the inverse of an orthogonal matrix is its transpose.

## The Lloyd-Max Algorithm: Optimal Scalar Quantization

Once every coordinate follows N(0, 1/d), we need the best possible scalar quantizer for this distribution. The **Lloyd-Max algorithm** finds it. It is essentially k-means clustering in one dimension, applied to the probability density function rather than to data samples.

A Lloyd-Max quantizer satisfies two optimality conditions simultaneously:

1. **Nearest-neighbor encoding**: Each input value maps to the nearest centroid. The decision boundaries are the midpoints between adjacent centroids.
2. **Centroid condition**: Each centroid equals the conditional expectation of the distribution within its cell:

`c_i = E[X | X in cell_i] = integral(x * f(x) dx, a_i, a_{i+1}) / integral(f(x) dx, a_i, a_{i+1})`

These two conditions define a fixed-point problem that the algorithm solves by alternating between them.

## Walking Through the Code

The `solve_lloyd_max()` function in `lloyd_max.py` (lines 32-86) implements the iterative procedure:

1. **Initialize centroids** uniformly in the range [-3.5 * sigma, 3.5 * sigma] where sigma = 1/sqrt(d). This covers approximately 99.95% of the Gaussian probability mass (line 52-53).
2. **Iterate** up to 200 times (line 55):
   - Compute decision **boundaries** as midpoints between adjacent centroids (line 57).
   - Update each **centroid** as the conditional expectation E[X | X in partition_i] using numerical integration with `scipy.integrate.quad` (lines 62-71). This is exact (up to numerical precision), not approximate.
   - Check **convergence**: stop if the maximum centroid shift is less than 1e-10 (lines 74-78).
3. Return the final centroids and boundaries as torch tensors (lines 83-86).

The exact distribution of each coordinate after random rotation of a d-dimensional unit vector is a Beta-type density supported on [-1, 1], implemented in `beta_pdf()` (lines 18-23):

`f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)`

For practical use, `gaussian_approx_pdf()` (lines 26-29) provides the N(0, 1/d) approximation that closely matches this exact density for d >= 64:

```python
def gaussian_approx_pdf(x, d):
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))
```

For d >= 64, the Gaussian approximation is highly accurate, and the resulting codebook is **symmetric around zero** -- because the underlying distribution is symmetric. This is verified in the test suite (`test_turboquant.py`, lines 31-35), where the sum of all centroids is checked to be near zero.

## The LloydMaxCodebook Class

The `LloydMaxCodebook` class (lines 107-131) wraps the solver into a reusable component:

- **Constructor**: Takes dimension d and bits b, calls `solve_lloyd_max()` to precompute centroids and boundaries. Also computes the expected distortion per coordinate analytically.
- **`quantize(x)`** (line 117): Maps each scalar value to the index of its nearest centroid using `argmin` over absolute differences.
- **`dequantize(indices)`** (line 123): Looks up centroid values from indices.

The codebook has 2^b centroids. At 3 bits, that is 8 centroids; at 2 bits, just 4. Despite this extreme compression, the Lloyd-Max codebook is provably optimal for the given distribution -- no other scalar quantizer with the same number of levels can achieve lower MSE.

## Why This Is Near-Optimal Vector Quantization

Here is the crucial insight that ties everything together. Scalar quantization is normally suboptimal for vectors because it ignores correlations between coordinates. But the random rotation *removes* those correlations, making coordinates approximately i.i.d. For i.i.d. coordinates, per-coordinate scalar quantization is near-optimal -- it achieves distortion close to the theoretical vector quantization lower bound.

The theoretical guarantee (from the TurboQuant paper, verified in `test_turboquant.py` line 63) is:

`MSE distortion per coordinate <= sqrt(3) * pi/2 * (1/4^b)`

This means the distortion drops by a factor of 4 for each additional bit. At 3 bits, the MSE is bounded by approximately 0.0425 per coordinate -- remarkably close to the information-theoretic limit.

The combination of random rotation + Lloyd-Max per-coordinate quantization achieves near-optimal vector quantization *without* the exponential codebook that full vector quantization would require, and *without* the per-block overhead that product quantization demands. It is the best of both worlds.

## Key Takeaways

- **Random orthogonal rotation makes vector coordinates approximately i.i.d.**, following N(0, 1/d), which enables a single fixed codebook to work for all coordinates without per-block normalization.
- **The Haar-distributed rotation matrix** is generated via QR decomposition of a Gaussian matrix with sign correction to ensure uniform distribution over the orthogonal group.
- **The Lloyd-Max algorithm finds the optimal scalar quantizer** for a known distribution by iterating between nearest-neighbor assignment and centroid updates via conditional expectations.
- **The codebook is symmetric, fixed, and data-independent** -- computed once from the known distribution, not calibrated on data samples.
- **Rotation + Lloyd-Max achieves near-optimal vector quantization distortion** (MSE proportional to 1/4^b) without exponential codebooks or per-block overhead.
