# Section 2: Quantization 101 -- Compression Without Catastrophic Loss

Section 1 established that the KV cache is the memory bottleneck. The natural follow-up question is: can we just store those vectors with fewer bits? The answer is yes -- but only if you are smart about how you do it. This section introduces quantization, the core technique behind all KV cache compression, and explains why the obvious approach (just round everything) leaves significant accuracy on the table.

## The JPEG Analogy

Think of quantization like compressing a photograph. A raw image off your camera might be 50 megabytes, but saved as a JPEG it drops to 2 megabytes -- and honestly, you can barely tell the difference. The JPEG encoder does not throw away pixels at random; it analyzes which visual details your eyes are least sensitive to and discards those first. Quantization does the same thing for numbers: instead of storing each value with 16 bits of precision (65,536 possible values), you round to 3 or 4 bits (8 or 16 possible values), choosing the rounding scheme carefully so that the reconstruction error is as small as possible.

In the context of the KV cache, each coordinate of a key or value vector is stored as an fp16 number (16 bits). If we can represent each coordinate with just 3 bits, we get roughly a 5x reduction in storage. The question is whether the attention mechanism -- which depends on precise inner products between query and key vectors -- can tolerate the rounding error.

## What Quantization Actually Does

Formally, quantization maps a continuous (or high-precision) value to one of a small, finite set of representative values called **centroids**. The full set of centroids is called a **codebook**:

`Codebook = {c_1, c_2, ..., c_K}   where K = 2^b for b bits`

For 3-bit quantization, K = 8 -- so every number gets mapped to one of 8 representative values. Instead of storing the original 16-bit number, you store only the index (3 bits) into the codebook. To reconstruct, you look up the centroid at that index.

The goal is to choose the codebook centroids so that the average reconstruction error is minimized. The standard objective is **Mean Squared Error (MSE)**:

`minimize E[||x - Q(x)||^2]`

where `Q(x)` is the quantized (reconstructed) version of x. This measures the average squared distance between original and reconstructed values -- the quantization equivalent of asking "how much did the compression change my data?"

## Scalar vs. Vector Quantization

There are two fundamentally different ways to apply quantization to a d-dimensional vector:

**Scalar quantization** treats each coordinate independently. You build a codebook of K scalar values and quantize each coordinate to its nearest centroid, one at a time. This is simple and fast, but it ignores the fact that coordinates may be correlated -- knowing that coordinate 1 is large tells you something about what coordinate 2 might be, and scalar quantization throws that information away.

**Vector quantization (VQ)** quantizes groups of coordinates together. Instead of a codebook of scalar centroids, you build a codebook of d-dimensional centroid *vectors*. The input vector is mapped to the nearest centroid vector in the codebook. This can capture correlations between coordinates and achieve better compression -- but the codebook grows exponentially with dimension. For d = 128 and 3-bit quantization, you would need 2^(3 x 128) = 2^384 centroid vectors, which is absurdly large.

**Product quantization (PQ)** is the practical middle ground. It splits the d-dimensional vector into M smaller sub-vectors (say, M = 16 groups of 8 dimensions each), and applies vector quantization independently to each sub-vector. This keeps codebook sizes manageable while still capturing some local correlations. PQ is the workhorse behind most practical vector compression systems -- but it has a hidden cost that we will examine in Section 3.

## Why Naive Uniform Quantization Fails

The simplest codebook would space centroids uniformly across the range of possible values -- like evenly spaced tick marks on a ruler. If your data is uniformly distributed, this is optimal. But KV cache activation vectors are not uniformly distributed.

After normalization layers (LayerNorm, RMSNorm) in transformers, the coordinates of activation vectors tend to follow an approximately **Gaussian (normal) distribution**. Most values cluster near zero, with progressively fewer values at larger magnitudes. This is a well-established empirical observation, and for the specific case of randomly rotated unit-norm vectors in d dimensions, it can be proven mathematically: each coordinate follows approximately N(0, 1/d) for d >= 64. More precisely, the exact distribution is a Beta-type density supported on [-1, 1] that converges to Gaussian as the dimension grows -- N(0, 1/d) is a highly accurate approximation, not an assumption, and this fact is what makes a fixed Gaussian-optimized codebook work so well.

Uniform quantization wastes its representation budget on this Gaussian data. It places the same number of centroids in the tails (where few values live) as it does near the center (where most values cluster). The result: too little resolution where you need it most, and wasted resolution where it barely matters.

The solution is to design the codebook specifically for the Gaussian distribution, placing more centroids near zero and fewer in the tails. This is exactly what the **Lloyd-Max algorithm** does -- and it is the foundation of TurboQuant's Stage 1, which we will cover in Section 4.

## Why This Matters for Inner Products

There is a subtlety that separates KV cache quantization from generic data compression. In most compression tasks, you care about reconstruction error: how close is the decompressed data to the original? For the KV cache, you care about something more specific: how close are the **inner products** `<q, k_quantized>` to the true inner products `<q, k>`?

An MSE-optimal codebook minimizes reconstruction error per coordinate, which is a good start. But minimizing per-coordinate error does not guarantee that inner products are preserved -- and worse, it does not guarantee that the inner product errors are *unbiased*. A biased compression scheme could systematically shift which tokens the model attends to, leading to degraded generation quality. TurboQuant addresses this with a two-stage approach: Stage 1 (MSE-optimal scalar quantization) gets the reconstruction close, and Stage 2 (QJL correction) eliminates the bias in inner products. But first, we need to understand why the classical approach to vector quantization introduces a hidden memory overhead that partially defeats its own purpose.

## Key Takeaways

- **Quantization maps high-precision values to a small codebook of representative centroids**, storing only the index (b bits) instead of the full value (16 bits).
- **The MSE objective** measures average squared reconstruction error -- the standard metric for codebook quality.
- **Scalar quantization** is simple but ignores correlations; **vector quantization** captures correlations but has exponential codebook growth; **product quantization** is the practical compromise.
- **Gaussian-distributed data requires non-uniform codebooks** -- uniform spacing wastes resolution near zero where most values cluster.
- **For KV cache compression, inner product accuracy matters more than per-coordinate accuracy**, and unbiased inner products are critical for correct attention patterns.
