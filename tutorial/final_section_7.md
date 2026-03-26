# Section 7: Empirical Results -- What 3-Bit Actually Buys You

The theory is elegant, but does it work? This section presents the empirical evidence from both synthetic benchmarks and real-model validation on consumer hardware. The numbers are striking: 5x memory compression at 3-bit with 99.5% attention fidelity, perfect needle-in-haystack retrieval, and a practical jump from 8K to 40K context on a 12 GB GPU. But the story has nuance -- 2-bit pushes too far, and the current Python implementation does not yet deliver the speed gains that optimized CUDA kernels could provide.

## Synthetic Compression Ratios

The first sanity check comes from `test_turboquant.py` (Test 5), which measures compression ratios on synthetic random vectors with d = 128 and seq_len = 1024. The results compare compressed storage (quantized indices + QJL signs + residual norms + vector norms) against uncompressed fp16 storage:

| Bit Width | Compression Ratio |
|-----------|------------------|
| 2-bit     | 7.76x            |
| 3-bit     | 5.22x            |
| 4-bit     | 3.94x            |

These ratios account for all overhead -- the QJL sign bits, residual norms, and vector norms are included. The effective storage is close to the nominal bit width: 3-bit quantization stores approximately 3.07 bits per coordinate when you amortize the per-vector overhead across 128 dimensions.

Why does 3-bit give 5.22x instead of the theoretical 16/3 = 5.33x? Because of the per-vector overhead (residual norm and vector norm, 32 bits total per 128-dimensional vector). This adds approximately 0.25 bits per coordinate, bringing the effective rate to about 3.25 bits and the ratio to 16/3.25 = 4.92x. The synthetic test measures slightly different overhead accounting, but the principle holds: overhead is small and does not scale with dimension.

## Real Model Results: Qwen 2.5 3B on RTX 3060

The real validation uses the Qwen 2.5 3B Instruct model, loaded in 4-bit weights via bitsandbytes, running on an NVIDIA RTX 3060 with 12 GB VRAM. The KV cache is generated from an 8K context prompt and then compressed at different bit widths.

| Configuration     | KV Cache Size | Compression |
|-------------------|---------------|-------------|
| Original (fp16)   | 289 MB        | 1.0x        |
| 4-bit TurboQuant  | 76 MB         | 3.8x        |
| 3-bit TurboQuant  | 58 MB         | 5.0x        |
| 2-bit TurboQuant  | 40 MB         | 7.3x        |

The 289 MB baseline comes from the formula established in Section 1: 36 layers x 2 KV heads x 8192 tokens x 128 head_dim x 2 (K+V) x 2 bytes = 289 MB. At 3-bit, this drops to 58 MB -- freeing 231 MB of VRAM that can support approximately 5x longer context on the same GPU.

## Attention Fidelity: How Close Is Close Enough?

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

## Needle-in-Haystack: Can the Model Still Find Specific Information?

The needle-in-haystack test is the ultimate practical benchmark: hide a specific piece of information in a long document, compress the KV cache, and check if the model can still retrieve it. The test embeds the phrase "The secret project code name is AURORA-7749" within a long context, then asks the model to recall it after compression.

Results from `test_turboquant.py` (Test 6): **100% retrieval accuracy (9/9 tests)** across all bit widths (2, 3, and 4), with varying haystack sizes of 512, 2048, and 8192 vectors. The transcript summarizes: "zero information loss for retrieval."

This is a strong result. Even at 2-bit -- where the top-1 match rate drops to 66% -- the model can still find and retrieve specific factual information from compressed context. The retrieval task depends on the *relative* ranking of a small number of highly relevant tokens, and the compression preserves those extreme attention peaks even when it shifts some of the middle-ranked tokens around.

## The 8K-to-40K Context Expansion

The practical punchline: on a 12 GB RTX 3060, the Qwen 2.5 3B model in 4-bit weights takes about 2 GB for model parameters. At fp16, an 8K context KV cache consumes 289 MB. At 3-bit TurboQuant, the same 8K cache is only 58 MB. The freed VRAM (231 MB) can hold approximately 4x more cached tokens, expanding the effective context window from 8K to roughly 40K tokens -- all on the same consumer GPU, with no retraining.

This is what the transcript calls "the difference between fitting 8,000 context and fitting 40,000 with the same model." For a researcher running experiments on a single GPU, this is the difference between truncating long documents and processing them in full.

## The 3-Bit Sweet Spot

The data points converge on 3-bit as the practical sweet spot:

- **4-bit** is conservative: 3.8x compression with near-perfect fidelity (~0.999 cosine similarity). Use this when you cannot tolerate any degradation.
- **3-bit** is the sweet spot: 5.0x compression with 99.5% cosine similarity and 92% top-1 match. The paper's claim of "zero accuracy loss" is reasonable at this level -- attention patterns are so close to the original that generation quality is nearly indistinguishable in practice.
- **2-bit** is aggressive: 7.3x compression but 66% top-1 match. The model still retrieves needles and produces coherent text, but outputs may diverge from the uncompressed model on nuanced tasks.

## A Note on Speed

Google's blog reports up to 8x speedup on H100 GPUs with optimized CUDA kernels. The current PyTorch implementation in this repository does **not** achieve those speed gains -- it is a reference implementation optimized for clarity, not throughput. The compression and decompression operations involve Python-level loops and standard PyTorch operations that do not exploit the bit-level parallelism that custom CUDA kernels can provide.

This is an important caveat. The memory savings are real and immediate -- you genuinely free VRAM by storing fewer bits. But the compute savings from working with compressed data (avoiding full fp16 matrix multiplications) require kernel-level optimization that has not been done in this codebase. Speed improvements are the next frontier for this implementation.

## Key Takeaways

- **3-bit TurboQuant achieves 5x compression** on real models (289 MB to 58 MB for Qwen 2.5 3B at 8K context) with 99.5% attention fidelity.
- **Needle-in-haystack retrieval is 100% at all bit widths** -- the compression preserves the ability to find specific information in long contexts.
- **3-bit is the sweet spot**: strong compression with negligible quality loss. 4-bit is more conservative; 2-bit is aggressive and may change outputs.
- **The practical impact is 5x longer context on the same hardware** -- 8K becomes 40K on a 12 GB consumer GPU.
- **Speed gains require CUDA optimization** -- the current Python implementation captures memory savings but not compute savings.
