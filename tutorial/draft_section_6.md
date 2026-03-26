# Section 6: The Full Estimator -- Asymmetric Attention Without Decompression

Sections 4 and 5 introduced the two stages of TurboQuant independently. Now we bring them together into the full estimator -- the formula that lets you compute attention scores directly from compressed keys, without ever decompressing them back to full precision. This is the core of TurboQuant's practical value: not just storing keys in fewer bits, but *using* them in fewer bits.

## The Complete Formula

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

## Why "Asymmetric"?

The estimator is called "asymmetric" because the two sides -- queries and keys -- are treated completely differently:

- **Keys** are compressed: rotated, quantized to (b-1)-bit indices, residual projected and sign-quantized, norm stored. This is done once when the key enters the KV cache.
- **Queries** are used at full precision: projected through S on the fly (`Sq = q @ S^T`) but never quantized. Each query is ephemeral -- used for one attention computation and then discarded.

This asymmetry is natural for the attention mechanism. In autoregressive generation, keys accumulate in the cache and persist for the entire conversation. Queries are generated fresh for each new token. Compressing the persistent data (keys) while keeping the ephemeral data (queries) at full precision gives maximum memory savings with minimum accuracy loss.

## Why the Estimator Is Unbiased

The combined estimator is **exactly unbiased**: `E[estimator] = <q, k>`.

The derivation is short:

1. Term 1 is deterministic: `<q, k_mse>` has no randomness (after quantization is fixed).
2. Term 2 has expectation `<q, r_k>` (proved in Section 5).
3. Therefore: `E[term1 + term2] = <q, k_mse> + <q, r_k> = <q, k_mse + r_k> = <q, k>`.

The unbiasedness holds because `k_mse + r_k = k` by definition of the residual. Stage 2 does not need to reconstruct r_k perfectly -- it only needs to provide an unbiased estimate of `<q, r_k>`, which the QJL trick guarantees.

This property is critical for attention. The softmax function that converts attention scores to weights is sensitive to systematic shifts. A biased estimator would consistently over- or under-weight certain tokens, degrading generation quality. An unbiased estimator means that on average, the attention pattern is exactly correct -- individual scores may have small random errors, but those errors do not accumulate in one direction.

## The Production Implementation: asymmetric_attention_scores()

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

## Why Values Use MSE-Only Compression

You might wonder: if QJL correction is so important for keys, why not use it for values too? The answer lies in how values are consumed in the attention computation.

The attention output is: `output = softmax(scores) @ V`

Values are combined via a **weighted average** -- the softmax weights sum to 1. When you average many vectors, individual per-vector errors tend to cancel out. If one value vector is slightly too large in some coordinate and another is slightly too small, the weighted sum averages out the errors. This is a direct consequence of the law of large numbers.

Keys, by contrast, participate through **inner products** that determine the relative ranking of tokens. A systematic bias in inner products would shift which tokens get the highest attention weights -- a categorical error that does not average out. This is why keys need the QJL correction (unbiased inner products) but values can get by with MSE-only reconstruction.

In the codebase, `TurboQuantCompressorMSE` (compressors.py, lines 161-221) handles value compression using only Stage 1 -- rotation, Lloyd-Max quantization, and MSE reconstruction. No QJL signs, no residual norms. This saves memory: values cost b bits per coordinate, while keys cost b bits per coordinate plus the QJL overhead.

## Key Takeaways

- **The full TurboQuant estimator combines two terms**: `<q, k_mse>` (MSE inner product) + `||r_k|| * sqrt(pi/2)/m * <Sq, sign(Sr_k)>` (QJL correction).
- **The estimator is exactly unbiased**: `E[estimator] = <q, k>`, because Stage 2 corrects the systematic error from Stage 1.
- **Attention scores are computed directly from compressed keys** -- no decompression step. The computation uses k_mse (fp16), sign bits (int8), and residual norms (fp16).
- **The design is asymmetric by necessity**: keys are compressed and cached long-term; queries stay at full precision and are used once.
- **Values use MSE-only compression** because the softmax-weighted average of values cancels per-vector errors, making QJL correction unnecessary and saving memory.
