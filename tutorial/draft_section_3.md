# Section 3: The Hidden Tax -- Memory Overhead in Classical Vector Quantization

Section 2 introduced product quantization (PQ) as the practical middle ground between scalar and full vector quantization. PQ splits a high-dimensional vector into sub-blocks, quantizes each independently, and achieves good compression. It sounds like the problem is solved -- until you look at the fine print. Every sub-block requires its own normalization constant, and those constants are stored at full precision. This "hidden tax" can add 1-2 extra bits per coordinate, partially defeating the purpose of compressing to 2-4 bits in the first place. Understanding this overhead is essential for appreciating what TurboQuant does differently.

## Why Sub-Blocks Need Scale Factors

Consider a 128-dimensional key vector that you want to compress with product quantization. A typical PQ setup might split it into M = 16 sub-vectors of 8 dimensions each. Before quantizing each sub-vector, you need to normalize it -- divide by its L2 norm or standard deviation -- so that the codebook centroids (which are calibrated for unit-scale data) can be applied.

The problem is that each sub-vector has a different scale. Sub-vector 1 might have a norm of 0.3 while sub-vector 2 has a norm of 1.7. If you quantize both using the same fixed codebook without normalizing first, the sub-vector with larger values will have its centroids too tightly packed (relatively), and the one with smaller values will have centroids too spread out. The quantization error would be terrible.

So you normalize each sub-vector, quantize the normalized version, and store the original norm alongside the indices so you can reconstruct the original scale later. This is the standard approach, and it works -- except that each of those normalization constants costs you 16 bits (fp16) or even 32 bits (fp32) of storage.

## Counting the Overhead

Let us do the arithmetic. With M = 16 sub-vectors from a 128-dimensional vector:

- **Quantized indices**: Each sub-vector uses b bits per coordinate, so the compressed data costs `b x 128` bits total.
- **Scale factors**: M = 16 normalization constants at 16 bits each = 256 bits total.
- **Overhead per coordinate**: 256 bits / 128 coordinates = 2 extra bits per coordinate.

If your target is 3-bit quantization (3 bits per coordinate for the actual data), the scale factors add 2 bits, bringing the effective bit rate to 5 bits per coordinate. You have achieved a 3.2x compression ratio instead of the 5.3x you were aiming for. At 2-bit quantization, the overhead is proportionally even worse: 2 bits of data plus 2 bits of overhead means you are effectively storing at 4 bits -- twice what you intended.

The Google Research blog puts it bluntly: "traditional vector quantization usually introduces its own 'memory overhead' as most methods require calculating and storing (in full precision) quantization constants for every small block of data. This overhead can add 1 or 2 extra bits per number, **partially defeating the purpose** of vector quantization."

## The Scale Factor Dilemma

You might think: just use fewer, larger sub-vectors. If you split 128 dimensions into only M = 4 sub-vectors of 32 dimensions each, the overhead drops to 4 x 16 / 128 = 0.5 bits per coordinate. Problem solved?

Not quite. Larger sub-vectors mean the codebook for each sub-vector must represent more complex patterns. For a b-bit codebook over 32-dimensional sub-vectors, you have 2^b centroid vectors -- only 8 centroids at 3-bit to cover all possible 32-dimensional patterns. The quantization quality degrades sharply as sub-vector dimension grows relative to codebook size. There is a fundamental trade-off: smaller sub-vectors give better quantization but worse overhead; larger sub-vectors reduce overhead but sacrifice quantization quality.

This is the dilemma that classical product quantization cannot escape. The overhead is intrinsic to the approach: if different sub-blocks of your vector have different magnitudes (and they always do), you *must* store that magnitude information somewhere, and that storage costs you bits.

## What TurboQuant Does Differently

TurboQuant sidesteps this dilemma entirely. Instead of splitting vectors into sub-blocks and normalizing each one, it applies a single global operation -- a random orthogonal rotation -- that makes *all* coordinates approximately identically distributed. After rotation, every coordinate follows approximately N(0, 1/d), regardless of the original vector's structure. A single fixed codebook, optimized once for this known distribution, works for every coordinate.

The result is that TurboQuant needs no per-sub-block normalization constants. The only per-vector overhead is:

- **One residual norm** (16 bits per vector for the QJL correction in Stage 2)
- **One vector norm** (16 bits per vector to preserve the original scale)

For a 128-dimensional vector, that is 32 bits of overhead across 128 coordinates -- just 0.25 bits per coordinate. Compare that to the 1-2 bits per coordinate that classical PQ requires. At 3-bit quantization, TurboQuant achieves a true effective bit rate close to 3.25 bits, while classical PQ might land at 4-5 bits effective.

This "zero memory overhead" property (zero in the sense that overhead does not scale with the number of sub-blocks or the dimension) is one of TurboQuant's key practical advantages. The rotation matrix and the Lloyd-Max codebook are both fixed -- computed once, stored once, and reused for every vector. They are not data-dependent, which means no calibration data is needed and no per-vector metadata beyond the two scalar norms.

## Key Takeaways

- **Classical product quantization requires per-sub-block scale factors** stored at full precision (fp16/fp32), adding 1-2 bits of overhead per coordinate.
- **This overhead partially defeats the purpose** of low-bit quantization: a target of 3 bits per coordinate can become 4-5 bits effective.
- **There is a fundamental trade-off** in PQ between sub-vector size (smaller = better quantization but more overhead) and overhead (fewer sub-vectors = less overhead but worse quantization).
- **TurboQuant eliminates per-sub-block overhead** by using a random rotation to make all coordinates identically distributed, enabling a single fixed codebook with no per-block normalization.
- **TurboQuant's per-vector overhead is just 0.25 bits per coordinate** (two fp16 scalars for a 128-dimensional vector), compared to 1-2 bits for classical methods.
