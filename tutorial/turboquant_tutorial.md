# TurboQuant: A Deep Technical Tutorial
### From KV Cache Bottlenecks to Near-Optimal Compression — Theory, Math, and Code

*For graduate students and researchers from any field who want to master TurboQuant in one session.*

*Based on: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026) and "Quantized Johnson-Lindenstrauss" (AISTATS 2026)*

---

## Table of Contents

1. [The Memory Wall — Why KV Cache is Your Bottleneck](#section-1)
2. [Quantization 101 — Compression Without Catastrophic Loss](#section-2)
3. [The Hidden Tax — Memory Overhead in Classical Vector Quantization](#section-3)
4. [Stage 1 — The Random Rotation Trick + Lloyd-Max Optimal Scalar Quantization](#section-4)
5. [Stage 2 — QJL: 1-Bit Johnson-Lindenstrauss Error Correction](#section-5)
6. [The Full Estimator — Asymmetric Attention Without Decompression](#section-6)
7. [Empirical Results — What 3-Bit Actually Buys You](#section-7)
8. [Code Walkthrough — Reading the PyTorch Implementation](#section-8)
9. [Researcher's Map — Implications, Open Questions, and What to Read Next](#section-9)

---

<a name="section-1"></a>

## Section 1: The Memory Wall -- Why KV Cache is Your Bottleneck

If you have ever tried to run an LLM with a long conversation on a consumer GPU, you have hit the memory wall. Not the model weights -- those you can quantize down to 4-bit and fit comfortably. The real killer is the **KV cache**, a data structure that most tutorials mention in passing but rarely explain as the binding constraint it actually is. Understanding why the KV cache grows so fast, and why that growth is the central problem TurboQuant solves, is the starting point for everything that follows.

### How Transformer Attention Works (The 30-Second Version)

Every transformer layer computes self-attention using three projections of each input token:

- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I carry?"

The attention formula is:

`Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`

where `d_k` is the head dimension (typically 128). For each new token the model generates, it must compare that token's query vector against every previous token's key to decide which past tokens matter most. Then it retrieves a weighted combination of their values.

This is the mechanism that gives transformers their power: every token can attend to every other token. But it comes with a cost.

### Why the KV Cache Exists

During autoregressive generation -- producing one token at a time -- naively recomputing K and V for all previous tokens at every step would be O(n^2) in computation. The **KV cache** eliminates this redundancy: it stores the K and V vectors from all previous tokens so they only need to be computed once. Each new token computes its own Q, K, V projections, appends K and V to the cache, and then uses the full cached K and V for attention.

Think of it as a high-speed digital cheat sheet. Instead of re-reading the entire conversation from scratch every time, the model keeps a running record of what it has already processed. This turns O(n^2) recomputation into O(n) per step. Efficient -- until the cheat sheet itself becomes the problem.

### How KV Cache Memory Grows

The KV cache memory scales as:

`Memory = layers x heads x seq_len x head_dim x 2 (K and V) x bytes_per_element`

Let us make this concrete with the **Qwen 2.5 3B** model (the model used in the TurboQuant validation experiments):

- 36 layers
- 2 KV heads per layer
- head_dim = 128
- fp16 precision (2 bytes per element)

For an 8K context window:

```
36 x 2 x 8192 x 128 x 2 x 2 bytes = 289 MB
```

That is 289 megabytes *just for the KV cache*. The model weights themselves, quantized to 4-bit with bitsandbytes, occupy about 2 GB. On a 12 GB RTX 3060, the KV cache quickly becomes the dominant memory consumer as context length grows.

And it gets worse linearly. Double the context to 16K, and the KV cache doubles to 578 MB. At 32K context, you are looking at over 1.1 GB. On consumer hardware, the KV cache -- not the model weights -- is what caps your maximum context length.

### Why This Matters: The 8K-to-40K Promise

Here is the punchline. If you could compress the KV cache by 5x -- from 289 MB down to 58 MB -- the VRAM you free up could support roughly 5x longer context. On that same 12 GB RTX 3060, an 8K context window becomes a 40K context window with the same model, the same GPU, and no retraining.

That is exactly what TurboQuant delivers at 3-bit quantization: 289 MB becomes 58 MB, a 5x compression ratio. The model still generates coherent, accurate text because the compression preserves 99.5% of the attention pattern (we will see the exact numbers in Section 7).

This is not a theoretical exercise. The KV cache is a high-dimensional vector store, and high-dimensional vectors are incredibly powerful but memory-hungry. Every K vector and every V vector across every layer and every head must be kept in GPU memory for the duration of the conversation. Compress those vectors intelligently, and you unlock dramatically longer contexts on existing hardware.

### The Core Constraint TurboQuant Addresses

The challenge is not just "make the numbers smaller." Naive compression -- say, rounding every fp16 value to its nearest 3-bit representation -- would destroy the subtle relationships between tokens that attention relies on. The inner products `Q @ K^T` that determine attention scores must remain accurate, or the model will attend to the wrong tokens and produce garbage.

TurboQuant's contribution is a compression scheme that achieves 5x memory reduction while keeping the inner product estimates **unbiased** -- meaning the compressed attention scores are, on average, exactly correct. How it accomplishes this is the subject of the next eight sections.

### Key Takeaways

- **The KV cache stores K and V vectors for all previous tokens** so the model does not recompute them at each generation step. It is essential for efficient autoregressive generation.
- **KV cache memory grows linearly with context length** and scales with layers, heads, and head dimension. For Qwen 2.5 3B at 8K context, this is 289 MB in fp16.
- **On consumer GPUs, the KV cache is the binding constraint on context length**, not the model weights (which can be aggressively quantized separately).
- **5x compression of the KV cache translates directly to 5x longer context** on the same hardware -- turning 8K into 40K on a 12 GB GPU.
- **The key challenge is compressing vectors without distorting inner products**, because attention scores depend on accurate Q-K dot products. TurboQuant solves this with an unbiased estimator.

---

<a name="section-2"></a>

## Section 2: Quantization 101 -- Compression Without Catastrophic Loss

Section 1 established that the KV cache is the memory bottleneck. The natural follow-up question is: can we just store those vectors with fewer bits? The answer is yes -- but only if you are smart about how you do it. This section introduces quantization, the core technique behind all KV cache compression, and explains why the obvious approach (just round everything) leaves significant accuracy on the table.

### The JPEG Analogy

Think of quantization like compressing a photograph. A raw image off your camera might be 50 megabytes, but saved as a JPEG it drops to 2 megabytes -- and honestly, you can barely tell the difference. The JPEG encoder does not throw away pixels at random; it analyzes which visual details your eyes are least sensitive to and discards those first. Quantization does the same thing for numbers: instead of storing each value with 16 bits of precision (65,536 possible values), you round to 3 or 4 bits (8 or 16 possible values), choosing the rounding scheme carefully so that the reconstruction error is as small as possible.

In the context of the KV cache, each coordinate of a key or value vector is stored as an fp16 number (16 bits). If we can represent each coordinate with just 3 bits, we get roughly a 5x reduction in storage. The question is whether the attention mechanism -- which depends on precise inner products between query and key vectors -- can tolerate the rounding error.

### What Quantization Actually Does

Formally, quantization maps a continuous (or high-precision) value to one of a small, finite set of representative values called **centroids**. The full set of centroids is called a **codebook**:

`Codebook = {c_1, c_2, ..., c_K}   where K = 2^b for b bits`

For 3-bit quantization, K = 8 -- so every number gets mapped to one of 8 representative values. Instead of storing the original 16-bit number, you store only the index (3 bits) into the codebook. To reconstruct, you look up the centroid at that index.

The goal is to choose the codebook centroids so that the average reconstruction error is minimized. The standard objective is **Mean Squared Error (MSE)**:

`minimize E[||x - Q(x)||^2]`

where `Q(x)` is the quantized (reconstructed) version of x. This measures the average squared distance between original and reconstructed values -- the quantization equivalent of asking "how much did the compression change my data?"

### Scalar vs. Vector Quantization

There are two fundamentally different ways to apply quantization to a d-dimensional vector:

**Scalar quantization** treats each coordinate independently. You build a codebook of K scalar values and quantize each coordinate to its nearest centroid, one at a time. This is simple and fast, but it ignores the fact that coordinates may be correlated -- knowing that coordinate 1 is large tells you something about what coordinate 2 might be, and scalar quantization throws that information away.

**Vector quantization (VQ)** quantizes groups of coordinates together. Instead of a codebook of scalar centroids, you build a codebook of d-dimensional centroid *vectors*. The input vector is mapped to the nearest centroid vector in the codebook. This can capture correlations between coordinates and achieve better compression -- but the codebook grows exponentially with dimension. For d = 128 and 3-bit quantization, you would need 2^(3 x 128) = 2^384 centroid vectors, which is absurdly large.

**Product quantization (PQ)** is the practical middle ground. It splits the d-dimensional vector into M smaller sub-vectors (say, M = 16 groups of 8 dimensions each), and applies vector quantization independently to each sub-vector. This keeps codebook sizes manageable while still capturing some local correlations. PQ is the workhorse behind most practical vector compression systems -- but it has a hidden cost that we will examine in Section 3.

### Why Naive Uniform Quantization Fails

The simplest codebook would space centroids uniformly across the range of possible values -- like evenly spaced tick marks on a ruler. If your data is uniformly distributed, this is optimal. But KV cache activation vectors are not uniformly distributed.

After normalization layers (LayerNorm, RMSNorm) in transformers, the coordinates of activation vectors tend to follow an approximately **Gaussian (normal) distribution**. Most values cluster near zero, with progressively fewer values at larger magnitudes. This is a well-established empirical observation, and for the specific case of randomly rotated unit-norm vectors in d dimensions, it can be proven mathematically: each coordinate follows approximately N(0, 1/d) for d >= 64. More precisely, the exact distribution is a Beta-type density supported on [-1, 1] that converges to Gaussian as the dimension grows -- N(0, 1/d) is a highly accurate approximation, not an assumption, and this fact is what makes a fixed Gaussian-optimized codebook work so well.

Uniform quantization wastes its representation budget on this Gaussian data. It places the same number of centroids in the tails (where few values live) as it does near the center (where most values cluster). The result: too little resolution where you need it most, and wasted resolution where it barely matters.

The solution is to design the codebook specifically for the Gaussian distribution, placing more centroids near zero and fewer in the tails. This is exactly what the **Lloyd-Max algorithm** does -- and it is the foundation of TurboQuant's Stage 1, which we will cover in Section 4.

### Why This Matters for Inner Products

There is a subtlety that separates KV cache quantization from generic data compression. In most compression tasks, you care about reconstruction error: how close is the decompressed data to the original? For the KV cache, you care about something more specific: how close are the **inner products** `<q, k_quantized>` to the true inner products `<q, k>`?

An MSE-optimal codebook minimizes reconstruction error per coordinate, which is a good start. But minimizing per-coordinate error does not guarantee that inner products are preserved -- and worse, it does not guarantee that the inner product errors are *unbiased*. A biased compression scheme could systematically shift which tokens the model attends to, leading to degraded generation quality. TurboQuant addresses this with a two-stage approach: Stage 1 (MSE-optimal scalar quantization) gets the reconstruction close, and Stage 2 (QJL correction) eliminates the bias in inner products. But first, we need to understand why the classical approach to vector quantization introduces a hidden memory overhead that partially defeats its own purpose.

### Key Takeaways

- **Quantization maps high-precision values to a small codebook of representative centroids**, storing only the index (b bits) instead of the full value (16 bits).
- **The MSE objective** measures average squared reconstruction error -- the standard metric for codebook quality.
- **Scalar quantization** is simple but ignores correlations; **vector quantization** captures correlations but has exponential codebook growth; **product quantization** is the practical compromise.
- **Gaussian-distributed data requires non-uniform codebooks** -- uniform spacing wastes resolution near zero where most values cluster.
- **For KV cache compression, inner product accuracy matters more than per-coordinate accuracy**, and unbiased inner products are critical for correct attention patterns.

---

<a name="section-3"></a>

## Section 3: The Hidden Tax -- Memory Overhead in Classical Vector Quantization

Section 2 introduced product quantization (PQ) as the practical middle ground between scalar and full vector quantization. PQ splits a high-dimensional vector into sub-blocks, quantizes each independently, and achieves good compression. It sounds like the problem is solved -- until you look at the fine print. Every sub-block requires its own normalization constant, and those constants are stored at full precision. This "hidden tax" can add 1-2 extra bits per coordinate, partially defeating the purpose of compressing to 2-4 bits in the first place. Understanding this overhead is essential for appreciating what TurboQuant does differently.

### Why Sub-Blocks Need Scale Factors

Consider a 128-dimensional key vector that you want to compress with product quantization. A typical PQ setup might split it into M = 16 sub-vectors of 8 dimensions each. Before quantizing each sub-vector, you need to normalize it -- divide by its L2 norm or standard deviation -- so that the codebook centroids (which are calibrated for unit-scale data) can be applied.

The problem is that each sub-vector has a different scale. Sub-vector 1 might have a norm of 0.3 while sub-vector 2 has a norm of 1.7. If you quantize both using the same fixed codebook without normalizing first, the sub-vector with larger values will have its centroids too tightly packed (relatively), and the one with smaller values will have centroids too spread out. The quantization error would be terrible.

So you normalize each sub-vector, quantize the normalized version, and store the original norm alongside the indices so you can reconstruct the original scale later. This is the standard approach, and it works -- except that each of those normalization constants costs you 16 bits (fp16) or even 32 bits (fp32) of storage.

### Counting the Overhead

Let us do the arithmetic. With M = 16 sub-vectors from a 128-dimensional vector:

- **Quantized indices**: Each sub-vector uses b bits per coordinate, so the compressed data costs `b x 128` bits total.
- **Scale factors**: M = 16 normalization constants at 16 bits each = 256 bits total.
- **Overhead per coordinate**: 256 bits / 128 coordinates = 2 extra bits per coordinate.

If your target is 3-bit quantization (3 bits per coordinate for the actual data), the scale factors add 2 bits, bringing the effective bit rate to 5 bits per coordinate. You have achieved a 3.2x compression ratio instead of the 5.3x you were aiming for. At 2-bit quantization, the overhead is proportionally even worse: 2 bits of data plus 2 bits of overhead means you are effectively storing at 4 bits -- twice what you intended.

The Google Research blog puts it bluntly: "traditional vector quantization usually introduces its own 'memory overhead' as most methods require calculating and storing (in full precision) quantization constants for every small block of data. This overhead can add 1 or 2 extra bits per number, **partially defeating the purpose** of vector quantization."

### The Scale Factor Dilemma

You might think: just use fewer, larger sub-vectors. If you split 128 dimensions into only M = 4 sub-vectors of 32 dimensions each, the overhead drops to 4 x 16 / 128 = 0.5 bits per coordinate. Problem solved?

Not quite. Larger sub-vectors mean the codebook for each sub-vector must represent more complex patterns. For a b-bit codebook over 32-dimensional sub-vectors, you have 2^b centroid vectors -- only 8 centroids at 3-bit to cover all possible 32-dimensional patterns. The quantization quality degrades sharply as sub-vector dimension grows relative to codebook size. There is a fundamental trade-off: smaller sub-vectors give better quantization but worse overhead; larger sub-vectors reduce overhead but sacrifice quantization quality.

This is the dilemma that classical product quantization cannot escape. The overhead is intrinsic to the approach: if different sub-blocks of your vector have different magnitudes (and they always do), you *must* store that magnitude information somewhere, and that storage costs you bits.

### What TurboQuant Does Differently

TurboQuant sidesteps this dilemma entirely. Instead of splitting vectors into sub-blocks and normalizing each one, it applies a single global operation -- a random orthogonal rotation -- that makes *all* coordinates approximately identically distributed. After rotation, every coordinate follows approximately N(0, 1/d), regardless of the original vector's structure. A single fixed codebook, optimized once for this known distribution, works for every coordinate.

The result is that TurboQuant needs no per-sub-block normalization constants. The only per-vector overhead is:

- **One residual norm** (16 bits per vector for the QJL correction in Stage 2)
- **One vector norm** (16 bits per vector to preserve the original scale)

For a 128-dimensional vector, that is 32 bits of overhead across 128 coordinates -- just 0.25 bits per coordinate. Compare that to the 1-2 bits per coordinate that classical PQ requires. At 3-bit quantization, TurboQuant achieves a true effective bit rate close to 3.25 bits, while classical PQ might land at 4-5 bits effective.

This "zero memory overhead" property (zero in the sense that overhead does not scale with the number of sub-blocks or the dimension) is one of TurboQuant's key practical advantages. The rotation matrix and the Lloyd-Max codebook are both fixed -- computed once, stored once, and reused for every vector. They are not data-dependent, which means no calibration data is needed and no per-vector metadata beyond the two scalar norms.

### Key Takeaways

- **Classical product quantization requires per-sub-block scale factors** stored at full precision (fp16/fp32), adding 1-2 bits of overhead per coordinate.
- **This overhead partially defeats the purpose** of low-bit quantization: a target of 3 bits per coordinate can become 4-5 bits effective.
- **There is a fundamental trade-off** in PQ between sub-vector size (smaller = better quantization but more overhead) and overhead (fewer sub-vectors = less overhead but worse quantization).
- **TurboQuant eliminates per-sub-block overhead** by using a random rotation to make all coordinates identically distributed, enabling a single fixed codebook with no per-block normalization.
- **TurboQuant's per-vector overhead is just 0.25 bits per coordinate** (two fp16 scalars for a 128-dimensional vector), compared to 1-2 bits for classical methods.

---

<a name="section-4"></a>

## Section 4: Stage 1 -- The Random Rotation Trick + Lloyd-Max Optimal Scalar Quantization

We now arrive at the first of TurboQuant's two stages. The idea is elegant: if you could make every coordinate of your vector follow the same known distribution, you could design one fixed codebook that works optimally for all of them -- no per-block normalization, no scale factors, no overhead. Stage 1 achieves this with two operations: a random orthogonal rotation that makes coordinates approximately independent and identically distributed, followed by the Lloyd-Max algorithm that builds the optimal scalar quantizer for that known distribution.

### Why Random Rotation Makes Quantization Work

Imagine a 128-dimensional key vector whose coordinates have wildly different variances. Some coordinates carry large values, others are near zero, and there may be correlations between them. If you apply per-coordinate scalar quantization directly, the codebook would need to accommodate all these different scales -- which is exactly the per-block normalization problem from Section 3.

Now multiply that vector by a random orthogonal matrix Pi. Something remarkable happens: the rotated vector's coordinates become approximately **independent and identically distributed (i.i.d.)**. Each coordinate follows approximately N(0, 1/d) for a unit-norm vector in d dimensions.

Why does this work? The key property is that if x has a rotationally symmetric distribution (like N(0, sigma^2 * I)), then Pi @ x has exactly the same distribution for any orthogonal Pi. For unit-norm vectors, random rotation effectively "spreads" the information uniformly across all coordinates. No single coordinate dominates; no pair is correlated. The mathematical foundation is that a random rotation drawn from the **Haar measure** (the uniform distribution over orthogonal matrices) is the unique transformation that achieves this uniformity.

After rotation, a single fixed codebook optimized for N(0, 1/d) works for every coordinate. No normalization constants needed. This is why Stage 1 eliminates the overhead that plagues classical product quantization.

### Generating the Rotation Matrix

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

### The Lloyd-Max Algorithm: Optimal Scalar Quantization

Once every coordinate follows N(0, 1/d), we need the best possible scalar quantizer for this distribution. The **Lloyd-Max algorithm** finds it. It is essentially k-means clustering in one dimension, applied to the probability density function rather than to data samples.

A Lloyd-Max quantizer satisfies two optimality conditions simultaneously:

1. **Nearest-neighbor encoding**: Each input value maps to the nearest centroid. The decision boundaries are the midpoints between adjacent centroids.
2. **Centroid condition**: Each centroid equals the conditional expectation of the distribution within its cell:

`c_i = E[X | X in cell_i] = integral(x * f(x) dx, a_i, a_{i+1}) / integral(f(x) dx, a_i, a_{i+1})`

These two conditions define a fixed-point problem that the algorithm solves by alternating between them.

### Walking Through the Code

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

### The LloydMaxCodebook Class

The `LloydMaxCodebook` class (lines 107-131) wraps the solver into a reusable component:

- **Constructor**: Takes dimension d and bits b, calls `solve_lloyd_max()` to precompute centroids and boundaries. Also computes the expected distortion per coordinate analytically.
- **`quantize(x)`** (line 117): Maps each scalar value to the index of its nearest centroid using `argmin` over absolute differences.
- **`dequantize(indices)`** (line 123): Looks up centroid values from indices.

The codebook has 2^b centroids. At 3 bits, that is 8 centroids; at 2 bits, just 4. Despite this extreme compression, the Lloyd-Max codebook is provably optimal for the given distribution -- no other scalar quantizer with the same number of levels can achieve lower MSE.

### Why This Is Near-Optimal Vector Quantization

Here is the crucial insight that ties everything together. Scalar quantization is normally suboptimal for vectors because it ignores correlations between coordinates. But the random rotation *removes* those correlations, making coordinates approximately i.i.d. For i.i.d. coordinates, per-coordinate scalar quantization is near-optimal -- it achieves distortion close to the theoretical vector quantization lower bound.

The theoretical guarantee (from the TurboQuant paper, verified in `test_turboquant.py` line 63) is:

`MSE distortion per coordinate <= sqrt(3) * pi/2 * (1/4^b)`

This means the distortion drops by a factor of 4 for each additional bit. At 3 bits, the MSE is bounded by approximately 0.0425 per coordinate -- remarkably close to the information-theoretic limit.

The combination of random rotation + Lloyd-Max per-coordinate quantization achieves near-optimal vector quantization *without* the exponential codebook that full vector quantization would require, and *without* the per-block overhead that product quantization demands. It is the best of both worlds.

### Key Takeaways

- **Random orthogonal rotation makes vector coordinates approximately i.i.d.**, following N(0, 1/d), which enables a single fixed codebook to work for all coordinates without per-block normalization.
- **The Haar-distributed rotation matrix** is generated via QR decomposition of a Gaussian matrix with sign correction to ensure uniform distribution over the orthogonal group.
- **The Lloyd-Max algorithm finds the optimal scalar quantizer** for a known distribution by iterating between nearest-neighbor assignment and centroid updates via conditional expectations.
- **The codebook is symmetric, fixed, and data-independent** -- computed once from the known distribution, not calibrated on data samples.
- **Rotation + Lloyd-Max achieves near-optimal vector quantization distortion** (MSE proportional to 1/4^b) without exponential codebooks or per-block overhead.

---

<a name="section-5"></a>

## Section 5: Stage 2 -- QJL: 1-Bit Johnson-Lindenstrauss Error Correction

Stage 1 gave us a near-optimal MSE reconstruction of each key vector. That is good enough if all you need is to get the vector back approximately. But for attention, you need something more specific: accurate **inner products** between queries and keys. Stage 1 alone introduces a systematic bias in those inner products -- the quantization error is correlated with the signal, so the errors do not cancel out. Stage 2 fixes this with a beautifully simple trick borrowed from dimensionality reduction theory: project the residual error through a random matrix, keep only the signs, and use those signs to build an unbiased correction term. The cost is just 1 extra bit per coordinate.

### The Residual: What Stage 1 Leaves Behind

After Stage 1 quantizes a key vector k into its MSE reconstruction k_mse, there is a residual:

`r = k - k_mse`

This residual is not random noise -- it is the structured error that Stage 1 could not capture. When you compute the inner product `<q, k_mse>` instead of the true `<q, k>`, you are missing the term `<q, r>`. If you could somehow recover this missing term cheaply, you could correct the inner product exactly. That is what QJL does.

### The Johnson-Lindenstrauss Lemma (Plain Language)

The Johnson-Lindenstrauss (JL) lemma is one of the most useful results in high-dimensional geometry. In plain language, it says: **random projections preserve inner products in expectation**.

More precisely, if S is an m x d random matrix with i.i.d. N(0,1) entries, then for any two vectors x and y:

`E[<Sx, Sy> / m] = <x, y>`

This means you can estimate the inner product of two high-dimensional vectors by projecting them into a lower-dimensional space and computing the inner product there. The projection introduces variance but no bias -- on average, the estimate is exactly right.

The practical power of JL is that the projection matrix S is random and data-independent. You do not need to know anything about x or y in advance. You just multiply by a random matrix and the inner product structure is preserved.

### The QJL Trick: Signs Are Enough

QJL (Quantized Johnson-Lindenstrauss) takes the JL idea one step further. Instead of storing the full projected vector Sr (which would cost m floating-point numbers), store only the **signs**:

`sign(Sr) in {-1, +1}^m`

This requires just 1 bit per projected dimension. Remarkably, these sign bits still carry enough information to estimate inner products. The key mathematical result is:

`E[sign(S_i^T r)] = sqrt(2/pi) * r / ||r||`

For a single row S_i of the projection matrix, the expected value of `sign(S_i^T r)` points in the direction of r, scaled by `sqrt(2/pi)`. This factor appears because `E[|Z|] = sqrt(2/pi)` for Z ~ N(0,1) -- taking the sign of a Gaussian projection loses magnitude information but preserves direction, and the `sqrt(2/pi)` quantifies exactly how much magnitude is lost.

### The Unbiased Estimator

Putting it together, the QJL correction term estimates `<q, r>` as follows:

`<q, r> ≈ ||r|| * sqrt(pi/2) / m * <Sq, sign(Sr)>`

where:
- `||r||` is the L2 norm of the residual (stored as one fp16 scalar per vector)
- `sqrt(pi/2) / m` is the correction scale that compensates for the sign quantization
- `Sq` is the full-precision projection of the query (NOT sign-quantized -- this is the asymmetric part)
- `sign(Sr)` is the sign-quantized projection of the residual (stored as 1-bit values)

The `sqrt(pi/2)` factor is the reciprocal of `sqrt(2/pi)` -- it undoes the magnitude loss introduced by taking signs. The division by m averages over the m projection dimensions.

**Why this is unbiased**: The proof follows directly from the sign expectation formula:

1. `E[sign(S_i^T r)] = sqrt(2/pi) * r_i_hat` where `r_i_hat = r / ||r||` is the unit direction of r
2. Therefore `E[<Sq, sign(Sr)>] = sqrt(2/pi) * m * <q, r> / ||r||`
3. Multiplying by `||r|| * sqrt(pi/2) / m` gives `E[correction] = <q, r>`

The correction exactly recovers the missing inner product term in expectation. Combined with Stage 1: `E[<q, k_mse> + correction] = <q, k_mse> + <q, r> = <q, k_mse + r> = <q, k>`.

### Walking Through the Code

The QJL matrix is generated by `generate_qjl_matrix()` in `turboquant.py` (lines 36-48):

```python
def generate_qjl_matrix(d, m=None, seed=None, device="cpu"):
    if m is None:
        m = d                          # Default: same dimensionality as input
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)  # i.i.d. N(0,1) entries
    return S.to(device)
```

The matrix S has shape (m, d) with m = d by default. Like the rotation matrix, it is random but fixed (seeded) and data-independent.

The inner product estimator lives in `TurboQuantProd.inner_product()` (turboquant.py, lines 165-192):

```python
def inner_product(self, y, compressed):
    # Term 1: inner product with MSE reconstruction
    x_mse = self.mse.dequantize(compressed["mse_indices"])
    term1 = (y * x_mse).sum(dim=-1)

    # Term 2: QJL correction
    y_projected = y @ self.S.T                              # Project query (NOT quantized)
    qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)

    m = self.qjl_dim
    correction_scale = math.sqrt(math.pi / 2) / m
    term2 = compressed["residual_norm"] * correction_scale * qjl_ip

    return term1 + term2
```

Notice the asymmetry: the query y is projected through S but kept at full precision (`y @ self.S.T`), while the key's residual was projected through S and sign-quantized during compression (`compressed["qjl_signs"]`). This is the "asymmetric" design -- queries are never compressed, only keys.

### Why Asymmetry Makes Sense

In autoregressive generation, each query is used once (for the current token), but each key is reused for every subsequent token's query. Compressing the keys (which are stored long-term in the KV cache) saves memory proportional to the sequence length. Compressing the queries would save almost nothing since they are ephemeral. The asymmetric design compresses exactly what needs compressing -- the cached keys -- while keeping queries at full precision for maximum accuracy.

### The Transcript's Analogy

The YouTube transcript offers a vivid way to think about the two stages: "Imagine compressing a photo and the colors shift slightly blue. Stage one is the compression. Stage two is the automatic color correction that perfectly undoes the shift -- and at the cost of just one extra bit per number."

Stage 1 compresses efficiently but introduces systematic error. Stage 2 adds a correction that is exactly unbiased, using only 1 bit of additional storage per coordinate. The test suite confirms this: Test 3 in `test_turboquant.py` (lines 74-113) verifies that the combined estimator has near-zero bias, while Test 4 (lines 116-144) shows that MSE-only inner products (without QJL) are systematically biased.

### Key Takeaways

- **The residual r = k - k_mse carries information about the true inner product** that Stage 1 alone misses, introducing bias in attention scores.
- **The Johnson-Lindenstrauss lemma** guarantees that random projections preserve inner products in expectation -- QJL exploits this by storing only the signs of the projected residual (1 bit each).
- **The correction scale sqrt(pi/2) / m** compensates exactly for the magnitude information lost by taking signs, making the combined estimator unbiased: `E[estimator] = <q, k>`.
- **The design is asymmetric**: queries stay at full precision while keys are compressed, matching the KV cache access pattern where keys are stored long-term and queries are ephemeral.
- **Stage 2 costs exactly 1 bit per coordinate** (the sign bits) plus one fp16 scalar per vector (the residual norm) -- a minimal price for eliminating inner product bias.

---

<a name="section-6"></a>

## Section 6: The Full Estimator -- Asymmetric Attention Without Decompression

Sections 4 and 5 introduced the two stages of TurboQuant independently. Now we bring them together into the full estimator -- the formula that lets you compute attention scores directly from compressed keys, without ever decompressing them back to full precision. This is the core of TurboQuant's practical value: not just storing keys in fewer bits, but *using* them in fewer bits.

### The Complete Formula

Combining Stage 1 (MSE reconstruction) and Stage 2 (QJL correction), the TurboQuant estimator for the inner product between a full-precision query q and a compressed key k is:

```
<q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2) / m * <Sq, sign(Sr_k)>
           \_term1_/   \__________________term2__________________/
```

where:
- `k_mse` is the Stage 1 MSE reconstruction of k (stored in fp16)
- `r_k = k - k_mse` is the residual from Stage 1 (not stored directly)
- `||r_k||` is the L2 norm of the residual (stored as one fp16 scalar)
- `S` is the fixed random Gaussian projection matrix (m x d)
- `sign(Sr_k)` is the sign-quantized projection of the residual (stored as 1-bit values)
- `Sq` is the full-precision projection of the query (computed on the fly)

**Term 1** is the straightforward inner product between the query and the MSE-reconstructed key. This is a standard matrix multiplication -- nothing special. It captures most of the true inner product but has a systematic bias because k_mse is not exactly k.

**Term 2** is the QJL correction. It estimates the missing `<q, r_k>` using the stored sign bits and residual norm. The `sqrt(pi/2) / m` factor compensates for the magnitude information lost by taking signs (as derived in Section 5). This term is unbiased, so it corrects the bias from Term 1 exactly in expectation.

### Why "Asymmetric"?

The estimator is called "asymmetric" because the two sides -- queries and keys -- are treated completely differently:

- **Keys** are compressed: rotated, quantized to (b-1)-bit indices, residual projected and sign-quantized, norm stored. This is done once when the key enters the KV cache.
- **Queries** are used at full precision: projected through S on the fly (`Sq = q @ S^T`) but never quantized. Each query is ephemeral -- used for one attention computation and then discarded.

This asymmetry is natural for the attention mechanism. In autoregressive generation, keys accumulate in the cache and persist for the entire conversation. Queries are generated fresh for each new token. Compressing the persistent data (keys) while keeping the ephemeral data (queries) at full precision gives maximum memory savings with minimum accuracy loss.

### Why the Estimator Is Unbiased

The combined estimator is **exactly unbiased**: `E[estimator] = <q, k>`.

The derivation is short:

1. Term 1 is deterministic: `<q, k_mse>` has no randomness (after quantization is fixed).
2. Term 2 has expectation `<q, r_k>` (proved in Section 5).
3. Therefore: `E[term1 + term2] = <q, k_mse> + <q, r_k> = <q, k_mse + r_k> = <q, k>`.

The unbiasedness holds because `k_mse + r_k = k` by definition of the residual. Stage 2 does not need to reconstruct r_k perfectly -- it only needs to provide an unbiased estimate of `<q, r_k>`, which the QJL trick guarantees.

This property is critical for attention. The softmax function that converts attention scores to weights is sensitive to systematic shifts. A biased estimator would consistently over- or under-weight certain tokens, degrading generation quality. An unbiased estimator means that on average, the attention pattern is exactly correct -- individual scores may have small random errors, but those errors do not accumulate in one direction.

### The Production Implementation: asymmetric_attention_scores()

The `TurboQuantCompressorV2` class in `compressors.py` (lines 25-158) implements the full pipeline for real transformer integration. The key method is `asymmetric_attention_scores()` (lines 122-158), which computes attention scores for a batch of queries against all compressed keys:

```python
def asymmetric_attention_scores(self, queries, compressed):
    k_mse = compressed["k_mse"].float()        # (B, H, S_k, D)
    signs = compressed["qjl_signs"].float()      # (B, H, S_k, D)
    r_norm = compressed["residual_norm"].float()  # (B, H, S_k)

    # Term 1: Q @ K_mse^T  (standard matmul)
    term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))

    # Term 2: QJL correction
    q_projected = torch.matmul(queries.float(), self.S.T)        # (B, H, S_q, D)
    qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))  # (B, H, S_q, S_k)

    m = self.S.shape[0]
    correction_scale = math.sqrt(math.pi / 2) / m
    term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)

    return term1 + term2
```

Notice how the implementation maps directly to the formula:

- **Term 1**: `Q @ K_mse^T` is a batch matrix multiplication -- the same operation as standard attention, just using the MSE-reconstructed keys instead of the original keys.
- **Term 2**: First project all queries through S (`Q @ S^T`), then compute inner products with all sign vectors (`q_projected @ signs^T`), then scale by `sqrt(pi/2) / m` and multiply by each key's residual norm.

The shapes tell the story. Term 1 produces (B, H, S_q, S_k) -- attention scores for every query-key pair. Term 2 produces the same shape. Adding them gives the corrected attention scores. At no point are the original keys decompressed -- the computation works entirely from the compressed representation {k_mse, signs, r_norm}.

### Why Values Use MSE-Only Compression

You might wonder: if QJL correction is so important for keys, why not use it for values too? The answer lies in how values are consumed in the attention computation.

The attention output is: `output = softmax(scores) @ V`

Values are combined via a **weighted average** -- the softmax weights sum to 1. When you average many vectors, individual per-vector errors tend to cancel out. If one value vector is slightly too large in some coordinate and another is slightly too small, the weighted sum averages out the errors. This is a direct consequence of the law of large numbers.

Keys, by contrast, participate through **inner products** that determine the relative ranking of tokens. A systematic bias in inner products would shift which tokens get the highest attention weights -- a categorical error that does not average out. This is why keys need the QJL correction (unbiased inner products) but values can get by with MSE-only reconstruction.

In the codebase, `TurboQuantCompressorMSE` (compressors.py, lines 161-221) handles value compression using only Stage 1 -- rotation, Lloyd-Max quantization, and MSE reconstruction. No QJL signs, no residual norms. This saves memory: values cost b bits per coordinate, while keys cost b bits per coordinate plus the QJL overhead.

### Key Takeaways

- **The full TurboQuant estimator combines two terms**: `<q, k_mse>` (MSE inner product) + `||r_k|| * sqrt(pi/2)/m * <Sq, sign(Sr_k)>` (QJL correction).
- **The estimator is exactly unbiased**: `E[estimator] = <q, k>`, because Stage 2 corrects the systematic error from Stage 1.
- **Attention scores are computed directly from compressed keys** -- no decompression step. The computation uses k_mse (fp16), sign bits (int8), and residual norms (fp16).
- **The design is asymmetric by necessity**: keys are compressed and cached long-term; queries stay at full precision and are used once.
- **Values use MSE-only compression** because the softmax-weighted average of values cancels per-vector errors, making QJL correction unnecessary and saving memory.

---

<a name="section-7"></a>

## Section 7: Empirical Results -- What 3-Bit Actually Buys You

The theory is elegant, but does it work? This section presents the empirical evidence from both synthetic benchmarks and real-model validation on consumer hardware. The numbers are striking: 5x memory compression at 3-bit with 99.5% attention fidelity, perfect needle-in-haystack retrieval, and a practical jump from 8K to 40K context on a 12 GB GPU. But the story has nuance -- 2-bit pushes too far, and the current Python implementation does not yet deliver the speed gains that optimized CUDA kernels could provide.

### Synthetic Compression Ratios

The first sanity check comes from `test_turboquant.py` (Test 5), which measures compression ratios on synthetic random vectors with d = 128 and seq_len = 1024. The results compare compressed storage (quantized indices + QJL signs + residual norms + vector norms) against uncompressed fp16 storage:

| Bit Width | Compression Ratio |
|-----------|------------------|
| 2-bit     | 7.76x            |
| 3-bit     | 5.22x            |
| 4-bit     | 3.94x            |

These ratios account for all overhead -- the QJL sign bits, residual norms, and vector norms are included. The effective storage is close to the nominal bit width: 3-bit quantization stores approximately 3.07 bits per coordinate when you amortize the per-vector overhead across 128 dimensions.

Why does 3-bit give 5.22x instead of the theoretical 16/3 = 5.33x? Because of the per-vector overhead (residual norm and vector norm, 32 bits total per 128-dimensional vector). This adds approximately 0.25 bits per coordinate, bringing the effective rate to about 3.25 bits and the ratio to 16/3.25 = 4.92x. The synthetic test measures slightly different overhead accounting, but the principle holds: overhead is small and does not scale with dimension.

### Real Model Results: Qwen 2.5 3B on RTX 3060

The real validation uses the Qwen 2.5 3B Instruct model, loaded in 4-bit weights via bitsandbytes, running on an NVIDIA RTX 3060 with 12 GB VRAM. The KV cache is generated from an 8K context prompt and then compressed at different bit widths.

| Configuration     | KV Cache Size | Compression |
|-------------------|---------------|-------------|
| Original (fp16)   | 289 MB        | 1.0x        |
| 4-bit TurboQuant  | 76 MB         | 3.8x        |
| 3-bit TurboQuant  | 58 MB         | 5.0x        |
| 2-bit TurboQuant  | 40 MB         | 7.3x        |

The 289 MB baseline comes from the formula established in Section 1: 36 layers x 2 KV heads x 8192 tokens x 128 head_dim x 2 (K+V) x 2 bytes = 289 MB. At 3-bit, this drops to 58 MB -- freeing 231 MB of VRAM that can support approximately 5x longer context on the same GPU.

### Attention Fidelity: How Close Is Close Enough?

Compression ratios are meaningless if the attention pattern is destroyed. The validation measures two metrics that quantify how faithfully the compressed attention matches the original:

**Cosine similarity** between compressed and uncompressed attention score vectors (across all key positions for each query):

| Bit Width | Cosine Similarity |
|-----------|------------------|
| 4-bit     | ~0.999 (99.9%)   |
| 3-bit     | 0.995 (99.5%)    |
| 2-bit     | significantly lower |

At 3-bit, the compressed attention scores point in almost exactly the same direction as the originals. A cosine similarity of 0.995 means the angle between the original and compressed score vectors is about 5.7 degrees -- nearly indistinguishable.

**Top-1 match rate** asks a more demanding question: does the model attend most strongly to the *same* token after compression?

| Bit Width | Top-1 Match Rate |
|-----------|-----------------|
| 3-bit     | 92%             |
| 2-bit     | 66%             |

At 3-bit, 92% of the time, the token that gets the highest attention weight is the same token that would get it without compression. A complementary metric, the **Top-5 match rate**, asks a more forgiving question: is the true top-1 token still among the compressed model's top 5? At 3-bit, this is also 92%, meaning that in the 8% of top-1 mismatches, the true winner has typically dropped just outside the top 5 -- but these cases involve tokens that were already close in attention score, so the impact on generation quality is minimal.

At 2-bit, the picture changes. A 66% top-1 match means the model would frequently attend to different tokens, which could noticeably alter outputs. As the transcript puts it: "the 66% top-one match means the model would sometimes attend to different tokens which could change outputs."

### Needle-in-Haystack: Can the Model Still Find Specific Information?

The needle-in-haystack test is the ultimate practical benchmark: hide a specific piece of information in a long document, compress the KV cache, and check if the model can still retrieve it. The test embeds the phrase "The secret project code name is AURORA-7749" within a long context, then asks the model to recall it after compression.

Results from `test_turboquant.py` (Test 6): **100% retrieval accuracy (9/9 tests)** across all bit widths (2, 3, and 4), with varying haystack sizes of 512, 2048, and 8192 vectors. The transcript summarizes: "zero information loss for retrieval."

This is a strong result. Even at 2-bit -- where the top-1 match rate drops to 66% -- the model can still find and retrieve specific factual information from compressed context. The retrieval task depends on the *relative* ranking of a small number of highly relevant tokens, and the compression preserves those extreme attention peaks even when it shifts some of the middle-ranked tokens around.

### The 8K-to-40K Context Expansion

The practical punchline: on a 12 GB RTX 3060, the Qwen 2.5 3B model in 4-bit weights takes about 2 GB for model parameters. At fp16, an 8K context KV cache consumes 289 MB. At 3-bit TurboQuant, the same 8K cache is only 58 MB. The freed VRAM (231 MB) can hold approximately 4x more cached tokens, expanding the effective context window from 8K to roughly 40K tokens -- all on the same consumer GPU, with no retraining.

This is what the transcript calls "the difference between fitting 8,000 context and fitting 40,000 with the same model." For a researcher running experiments on a single GPU, this is the difference between truncating long documents and processing them in full.

### The 3-Bit Sweet Spot

The data points converge on 3-bit as the practical sweet spot:

- **4-bit** is conservative: 3.8x compression with near-perfect fidelity (~0.999 cosine similarity). Use this when you cannot tolerate any degradation.
- **3-bit** is the sweet spot: 5.0x compression with 99.5% cosine similarity and 92% top-1 match. The paper's claim of "zero accuracy loss" is reasonable at this level -- attention patterns are so close to the original that generation quality is nearly indistinguishable in practice.
- **2-bit** is aggressive: 7.3x compression but 66% top-1 match. The model still retrieves needles and produces coherent text, but outputs may diverge from the uncompressed model on nuanced tasks.

### A Note on Speed

Google's blog reports up to 8x speedup on H100 GPUs with optimized CUDA kernels. The current PyTorch implementation in this repository does **not** achieve those speed gains -- it is a reference implementation optimized for clarity, not throughput. The compression and decompression operations involve Python-level loops and standard PyTorch operations that do not exploit the bit-level parallelism that custom CUDA kernels can provide.

This is an important caveat. The memory savings are real and immediate -- you genuinely free VRAM by storing fewer bits. But the compute savings from working with compressed data (avoiding full fp16 matrix multiplications) require kernel-level optimization that has not been done in this codebase. Speed improvements are the next frontier for this implementation.

### Key Takeaways

- **3-bit TurboQuant achieves 5x compression** on real models (289 MB to 58 MB for Qwen 2.5 3B at 8K context) with 99.5% attention fidelity.
- **Needle-in-haystack retrieval is 100% at all bit widths** -- the compression preserves the ability to find specific information in long contexts.
- **3-bit is the sweet spot**: strong compression with negligible quality loss. 4-bit is more conservative; 2-bit is aggressive and may change outputs.
- **The practical impact is 5x longer context on the same hardware** -- 8K becomes 40K on a 12 GB consumer GPU.
- **Speed gains require CUDA optimization** -- the current Python implementation captures memory savings but not compute savings.

---

<a name="section-8"></a>

## Section 8: Code Walkthrough -- Reading the PyTorch Implementation

The previous sections developed the theory; this section grounds it in code. We will walk through the four core files in the TurboQuant repository, tracing how each mathematical concept maps to a specific class, method, or line of code. The goal is not to explain every line, but to give you a reading map so you can navigate the codebase confidently and understand the design decisions behind the implementation.

### File Structure

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

### lloyd_max.py: The Foundation

This file contains the Lloyd-Max codebook solver -- the algorithm that finds the optimal scalar quantizer for a known distribution (Section 4).

**`beta_pdf(x, d)`** (lines 18-23) implements the exact distribution of each coordinate after random rotation of a d-dimensional unit vector. **`gaussian_approx_pdf(x, d)`** (lines 26-29) provides the N(0, 1/d) Gaussian approximation used in practice.

**`solve_lloyd_max(d, bits, pdf_func)`** (lines 32-86) is the iterative solver. The key design choice is on lines 52-53: centroids are initialized uniformly in [-3.5 * sigma, 3.5 * sigma] where sigma = 1/sqrt(d). This range covers 99.95% of the Gaussian mass, ensuring the solver starts with good coverage. The iteration loop (line 55) alternates between computing boundaries as midpoints (line 57) and updating centroids as conditional expectations via `scipy.integrate.quad` (lines 62-71). Convergence is checked against a tolerance of 1e-10 (line 74) -- this is tight, reflecting the fact that the codebook is computed once and reused forever.

**`LloydMaxCodebook`** (lines 107-131) wraps the solver into a PyTorch module. Two methods matter:

- **`quantize(x)`** (line 117): Maps each scalar to its nearest centroid index via `argmin` over absolute differences with all centroids. This is a brute-force nearest-neighbor search, which is fine because the codebook has at most 2^4 = 16 entries.
- **`dequantize(indices)`** (line 123): Looks up centroid values from indices -- a simple table lookup.

The class also precomputes `self.distortion` -- the expected MSE per coordinate, computed analytically by `compute_expected_distortion()` (lines 89-104). This number is used in tests to verify that empirical distortion matches theory.

### turboquant.py: The Core Engine

This file contains the three main classes that implement TurboQuant's two-stage pipeline.

#### generate_rotation_matrix() (lines 18-33)

Generates a Haar-distributed random orthogonal matrix via QR decomposition of a Gaussian matrix, with the sign correction on the diagonal of R to ensure true uniform distribution over the orthogonal group (covered in detail in Section 4). The seed parameter ensures reproducibility -- the same seed always produces the same rotation matrix.

#### generate_qjl_matrix() (lines 36-48)

Generates the random Gaussian projection matrix S for QJL. The default is m = d (same number of projections as input dimensions). Like the rotation matrix, it is seeded for reproducibility. The matrix entries are i.i.d. N(0,1) -- no normalization or orthogonalization is applied, because the JL guarantee works with raw Gaussian matrices.

#### TurboQuantMSE (lines 51-100)

This class implements Stage 1: random rotation + Lloyd-Max per-coordinate quantization. It is the MSE-only quantizer used for value vectors.

**Constructor** (lines 57-69): The key line is `self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))` (line 64). The rotation matrix is stored as a buffer -- persistent, device-aware, but not a learnable parameter. The `LloydMaxCodebook` is created with the specified dimension and bit width.

**`rotate(x)`** and **`unrotate(y)`** (lines 71-78): Rotation is `x @ Pi^T` and un-rotation is `y @ Pi`. Since Pi is orthogonal, Pi^(-1) = Pi^T, so these are exact inverses. Note the convention: x is a row vector (batch of row vectors), so rotation multiplies on the right by Pi^T rather than on the left by Pi.

**`quantize(x)`** (line 80) and **`dequantize(indices)`** (line 88) contain the actual work. `quantize()` rotates the input and finds the nearest centroid index per coordinate. `dequantize()` looks up centroid values and unrotates back to the original space. **`forward(x)`** (line 93) simply delegates to both: `indices = self.quantize(x)` followed by `x_hat = self.dequantize(indices)`. The conceptual pipeline across these methods is four steps:

1. Rotate: `y = x @ Pi^T`
2. Quantize: find nearest centroid index for each coordinate
3. Dequantize: look up centroid values
4. Unrotate: `x_hat = centroids @ Pi`

The method returns both the reconstructed vector `x_hat` and the quantization indices. The indices are what gets stored; the reconstruction is computed on the fly when needed.

#### TurboQuantProd (lines 103-196)

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

### compressors.py: Production Integration

This file bridges the gap between the core quantizers and real transformer models. It handles 4D tensor shapes (batch, heads, sequence, head_dim), explicit normalization, and the asymmetric attention score computation.

#### TurboQuantCompressorV2 (lines 25-158)

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

#### TurboQuantCompressorMSE (lines 161-221)

The value compressor. Simpler than V2 because values do not need QJL correction. It stores quantization indices (uint8) and vector norms (fp16), and provides a `decompress()` method that reconstructs value vectors by dequantizing, unrotating, and rescaling. No asymmetric attention method here -- values are fully decompressed before being used in the softmax-weighted sum.

#### Key Design Choice: Keys vs. Values

The split between `TurboQuantCompressorV2` (keys) and `TurboQuantCompressorMSE` (values) reflects the different requirements derived in Section 6:

- **Keys** need unbiased inner products for attention scores, so they use the full two-stage pipeline with QJL correction. The trade-off: keys store k_mse (fp16) + signs (int8) + residual norm (fp16).
- **Values** need good MSE reconstruction for the weighted average, so they use Stage 1 only. The trade-off: values store indices (uint8) + vector norms (fp16) -- less memory per vector.

This asymmetry is a principled design decision, not a shortcut. It allocates the QJL overhead (sign bits + residual norms) only where it is mathematically necessary.

### Key Takeaways

- **The codebase is four files deep**: `lloyd_max.py` (codebook solver) -> `turboquant.py` (core quantizers) -> `compressors.py` (production wrappers).
- **`LloydMaxCodebook`** precomputes the optimal codebook once; `quantize()` and `dequantize()` are simple nearest-neighbor and table-lookup operations.
- **`TurboQuantMSE`** stores the rotation matrix as a buffer and implements rotate/quantize/dequantize/unrotate as its forward pass.
- **`TurboQuantProd`** uses (b-1) bits for MSE + 1 bit for QJL signs, with `inner_product()` implementing the asymmetric estimator.
- **`TurboQuantCompressorV2`** handles the full pipeline for real models: normalize, rotate, quantize, compute k_mse, compute residual, project and sign-quantize, then compute asymmetric attention scores directly from compressed data.

---

<a name="section-9"></a>

## Section 9: Researcher's Map -- Implications, Open Questions, and What to Read Next

You now understand what TurboQuant does, why it works, and how to read its code. This final section zooms out. Where does TurboQuant fit in the broader landscape? What problems does it solve beyond KV cache compression? What does it *not* solve? And if you want to go deeper, what should you read and try first?

### Beyond KV Cache: Vector Search and Maximum Inner Product Search

The math behind TurboQuant -- random rotation, per-coordinate quantization, and QJL-based inner product estimation -- has nothing inherently to do with transformers or attention. The core problem it solves is: **compress high-dimensional vectors so that inner products can be computed efficiently from the compressed representation, without bias.**

This is exactly the problem that arises in **vector search**, also known as Maximum Inner Product Search (MIPS) or Approximate Nearest Neighbor (ANN) search. In these settings, you have a database of millions (or billions) of vectors -- document embeddings, image features, user preference vectors -- and you need to find the ones most similar to a query vector. The bottleneck is the same as in KV cache: storing and scanning all those vectors is expensive.

TurboQuant's approach -- rotation to make coordinates i.i.d., fixed Lloyd-Max codebook, QJL correction for unbiased inner products -- can be applied directly to vector search. The asymmetric design (compress database vectors, keep query at full precision) is natural for search, where the database is indexed once and queried many times. The blog confirms this broader applicability: TurboQuant "demonstrates a transformative shift in high-dimensional search" by setting "a new benchmark for achievable speed" with "near-optimal distortion rates in a data-oblivious manner."

If you work on retrieval-augmented generation (RAG), embedding-based search, or recommendation systems, TurboQuant's techniques are directly relevant to your vector index compression strategy.

### What TurboQuant Does NOT Solve

It is equally important to understand the boundaries of TurboQuant's contribution:

**Weight quantization** is a different problem. TurboQuant compresses activation vectors (keys and values computed at inference time), not model weights. Weight quantization methods like GPTQ, AWQ, and bitsandbytes nf4 address a complementary challenge with different constraints -- weights are static, known at compile time, and used in matrix multiplications with different numerical sensitivity. TurboQuant and weight quantization are orthogonal: you can (and should) use both simultaneously. The validation experiments use a 4-bit weight-quantized model with TurboQuant on the KV cache.

**Prefill speed** is not addressed. TurboQuant compresses the KV cache during the decode phase (autoregressive generation), but the initial prefill -- processing the entire prompt to populate the KV cache -- still runs at normal speed. For workloads dominated by long-prompt prefill rather than long-context generation, TurboQuant helps with memory but not with the compute bottleneck.

**Throughput vs. latency** requires CUDA optimization. The current Python implementation achieves the memory savings (fewer bits stored) but does not achieve the compute speedups that Google reports with optimized CUDA kernels (up to 8x on H100). The asymmetric attention score computation involves additional matrix multiplications (projecting queries through S, computing the QJL correction term) that add compute overhead in an unoptimized implementation. Realizing the speed benefits requires custom CUDA kernels that can exploit bit-level parallelism and fused operations -- this is engineering work that has not been done in this repository.

**Accuracy at extreme compression** has limits. As Section 7 showed, 2-bit quantization achieves impressive compression (7.3x) but drops the top-1 attention match to 66%. For tasks requiring precise factual recall or reasoning over subtle distinctions, this may not be acceptable. TurboQuant does not claim to solve the fundamental information-theoretic limits of lossy compression -- it operates near those limits, but the limits still exist.

### The Frontier: CUDA Kernel Optimization

The most impactful next step for this implementation is **CUDA kernel optimization**. The current PyTorch code uses standard tensor operations -- matrix multiplications, argmin searches, sign functions -- that do not exploit the structure of the compressed data.

A custom CUDA kernel could:
- Bit-pack the QJL signs (currently stored as int8, wasting 7 of 8 bits per element)
- Fuse the rotation + quantization + residual computation into a single kernel pass
- Fuse the two-term attention score computation (term1 + term2) to avoid materializing intermediate tensors
- Exploit the fact that codebook lookups are simple table reads, which map well to GPU shared memory

Google's reported numbers (up to 8x speedup on H100) suggest that the gap between reference implementation and optimized implementation is substantial. For anyone with CUDA kernel development experience, closing this gap is a high-impact contribution.

### Open Questions

Several research questions remain open:

**Does TurboQuant compose with other compression methods?** For example, can you apply TurboQuant to the KV cache and simultaneously use KV cache eviction (dropping old tokens) or sparse attention patterns? The mathematical properties (unbiasedness, near-optimal distortion) should hold regardless, but the interaction effects on end-to-end generation quality have not been studied.

**How does TurboQuant perform on Mixture-of-Experts (MoE) models?** MoE models have different activation patterns -- only a subset of experts are active for each token, leading to different statistical properties in the KV cache vectors. Whether the N(0, 1/d) distribution assumption holds for MoE activations, and whether the compression ratios transfer, is an open empirical question.

**Can the QJL projection dimension m be reduced below d?** The default is m = d (128 projections for 128-dimensional vectors), which means the QJL signs cost 1 bit per coordinate. If m could be reduced to d/2 or d/4 with acceptable accuracy loss, the storage overhead would drop further. The trade-off between projection dimension and estimator variance is well-characterized theoretically but not fully explored empirically for KV cache applications.

**What is the interaction with different attention mechanisms?** Grouped-query attention (GQA), multi-query attention (MQA), and sliding-window attention all modify how keys and values are stored and accessed. TurboQuant's asymmetric design should adapt naturally, but the optimal bit allocation may differ across architectures.

### Papers to Read (In Order)

If you want to go deeper into the theory and context, here are five papers in a recommended reading order:

1. **Johnson & Lindenstrauss (1984)**: "Extensions of Lipschitz mappings into a Hilbert space." The original JL lemma -- the foundation for random projection methods. Short and elegant.

2. **Jegou, Douze & Schmid (2011)**: "Product Quantization for Nearest Neighbor Search." The PQ method that became the standard for vector compression. Understanding PQ's strengths and overhead problem (Section 3) makes TurboQuant's contribution clearer.

3. **QJL paper (arXiv 2502.02617)**: The Quantized Johnson-Lindenstrauss paper that introduces the 1-bit sign projection trick and proves its unbiasedness. This is the theoretical engine behind Stage 2.

4. **TurboQuant paper (arXiv 2504.19874, ICLR 2026)**: The full TurboQuant paper with theoretical analysis, distortion bounds, and experimental results on production-scale models.

5. **KIVI**: The KV cache quantization baseline that TurboQuant improves upon. Reading KIVI helps you understand what the prior state of the art looked like and where TurboQuant's improvements are most significant.

### Three Things to Try

Theory is best cemented by practice. Here are three concrete exercises, ordered by difficulty:

1. **Run `test_turboquant.py` to reproduce the distortion results.** All seven tests should pass. Pay attention to Test 3 (unbiased inner products), Test 4 (MSE-only is biased -- the contrast), and Test 5 (compression ratios). This takes minutes and requires only PyTorch and scipy.

2. **Swap `TurboQuantKVCache` into a HuggingFace model's attention layer.** The `TurboQuantKVCache` class in `turboquant.py` (lines 199-286) is designed as a drop-in concept for a standard KV cache. Pick a small model (GPT-2, Pythia-70M), intercept the attention layer's key-value storage, and route it through TurboQuantKVCache. Compare generation quality with and without compression at different bit widths.

3. **Experiment with different bit widths on your own model.** Run `validate.py` with a model you care about (modify `MODEL_NAME` on line 18). Try 2-bit, 3-bit, and 4-bit. Measure the attention fidelity metrics (cosine similarity, top-1 match) for your specific model and compare against the Qwen 2.5 3B numbers from Section 7. Different models have different sensitivity to KV cache quantization -- finding the sweet spot for your use case is valuable empirical knowledge.

### Key Takeaways

- **TurboQuant's math applies beyond KV cache** to any setting that requires compressed inner products: vector search, MIPS, embedding retrieval, recommendation systems.
- **TurboQuant does not solve weight quantization, prefill speed, or compute throughput** -- it is specifically a memory compression method for cached activation vectors.
- **CUDA kernel optimization is the highest-impact engineering frontier** -- the gap between reference implementation and optimized kernels is where the speed gains live.
- **Open research questions include** composability with other compression methods, MoE model applicability, reduced projection dimensions, and interaction with non-standard attention mechanisms.
- **Start with the tests, then try integration, then experiment** -- the codebase is small enough to understand fully in a single session.

---

*Tutorial produced by an agent team (researcher -> writer -> editor -> assembler) using Claude Code.*
*Source implementation: https://github.com/tonbistudio/turboquant-pytorch*
