# Section 1: The Memory Wall -- Why KV Cache is Your Bottleneck

If you have ever tried to run an LLM with a long conversation on a consumer GPU, you have hit the memory wall. Not the model weights -- those you can quantize down to 4-bit and fit comfortably. The real killer is the **KV cache**, a data structure that most tutorials mention in passing but rarely explain as the binding constraint it actually is. Understanding why the KV cache grows so fast, and why that growth is the central problem TurboQuant solves, is the starting point for everything that follows.

## How Transformer Attention Works (The 30-Second Version)

Every transformer layer computes self-attention using three projections of each input token:

- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I carry?"

The attention formula is:

`Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`

where `d_k` is the head dimension (typically 128). For each new token the model generates, it must compare that token's query vector against every previous token's key to decide which past tokens matter most. Then it retrieves a weighted combination of their values.

This is the mechanism that gives transformers their power: every token can attend to every other token. But it comes with a cost.

## Why the KV Cache Exists

During autoregressive generation -- producing one token at a time -- naively recomputing K and V for all previous tokens at every step would be O(n^2) in computation. The **KV cache** eliminates this redundancy: it stores the K and V vectors from all previous tokens so they only need to be computed once. Each new token computes its own Q, K, V projections, appends K and V to the cache, and then uses the full cached K and V for attention.

Think of it as a high-speed digital cheat sheet. Instead of re-reading the entire conversation from scratch every time, the model keeps a running record of what it has already processed. This turns O(n^2) recomputation into O(n) per step. Efficient -- until the cheat sheet itself becomes the problem.

## How KV Cache Memory Grows

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

## Why This Matters: The 8K-to-40K Promise

Here is the punchline. If you could compress the KV cache by 5x -- from 289 MB down to 58 MB -- the VRAM you free up could support roughly 5x longer context. On that same 12 GB RTX 3060, an 8K context window becomes a 40K context window with the same model, the same GPU, and no retraining.

That is exactly what TurboQuant delivers at 3-bit quantization: 289 MB becomes 58 MB, a 5x compression ratio. The model still generates coherent, accurate text because the compression preserves 99.5% of the attention pattern (we will see the exact numbers in Section 7).

This is not a theoretical exercise. The KV cache is a high-dimensional vector store, and high-dimensional vectors are incredibly powerful but memory-hungry. Every K vector and every V vector across every layer and every head must be kept in GPU memory for the duration of the conversation. Compress those vectors intelligently, and you unlock dramatically longer contexts on existing hardware.

## The Core Constraint TurboQuant Addresses

The challenge is not just "make the numbers smaller." Naive compression -- say, rounding every fp16 value to its nearest 3-bit representation -- would destroy the subtle relationships between tokens that attention relies on. The inner products `Q @ K^T` that determine attention scores must remain accurate, or the model will attend to the wrong tokens and produce garbage.

TurboQuant's contribution is a compression scheme that achieves 5x memory reduction while keeping the inner product estimates **unbiased** -- meaning the compressed attention scores are, on average, exactly correct. How it accomplishes this is the subject of the next eight sections.

## Key Takeaways

- **The KV cache stores K and V vectors for all previous tokens** so the model does not recompute them at each generation step. It is essential for efficient autoregressive generation.
- **KV cache memory grows linearly with context length** and scales with layers, heads, and head dimension. For Qwen 2.5 3B at 8K context, this is 289 MB in fp16.
- **On consumer GPUs, the KV cache is the binding constraint on context length**, not the model weights (which can be aggressively quantized separately).
- **5x compression of the KV cache translates directly to 5x longer context** on the same hardware -- turning 8K into 40K on a 12 GB GPU.
- **The key challenge is compressing vectors without distorting inner products**, because attention scores depend on accurate Q-K dot products. TurboQuant solves this with an unbiased estimator.
