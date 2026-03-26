# TurboQuant PyTorch — Implementation + Deep Tutorial

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — **enhanced with a comprehensive, research-grade tutorial** that teaches the theory, math, and code from the ground up.

> **This repo is a clone of [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)**, extended with a full tutorial suite produced by an AI agent team. The original implementation and all experimental results are by Tonbi Studio. See [Acknowledgements](#acknowledgements) below.

---

## What's New in This Fork

The original repo provides the implementation and validation results. This fork adds:

### 📖 [`tutorial/turboquant_tutorial.md`](tutorial/turboquant_tutorial.md) — The Deep Tutorial

A ~9,500-word, nine-section technical tutorial written for graduate students and researchers from any field who want to master TurboQuant in one session. It covers:

| Section | Topic |
|---|---|
| 1 | **The Memory Wall** — Why KV cache is the real bottleneck, with concrete memory formulas and hardware numbers |
| 2 | **Quantization 101** — Codebooks, MSE objective, scalar vs vector quantization, why Gaussian distributions matter |
| 3 | **The Hidden Tax** — Memory overhead in classical VQ, why per-block scale factors add 1-2 bits, and how TurboQuant eliminates them |
| 4 | **Stage 1: Random Rotation + Lloyd-Max** — Haar-distributed orthogonal matrices, Lloyd-Max optimality conditions with equations, code walkthrough |
| 5 | **Stage 2: QJL** — The Johnson-Lindenstrauss lemma, the 1-bit sign estimator, the √(π/2)/m correction factor, and the unbiasedness proof |
| 6 | **The Full Estimator** — The two-stage formula assembled: `⟨q,k⟩ ≈ ⟨q,k_mse⟩ + ‖r_k‖·√(π/2)/m·⟨Sq, sign(Sr_k)⟩`, asymmetric attention without decompression |
| 7 | **Empirical Results** — All compression and fidelity numbers from real experiments on Qwen 2.5 3B + RTX 3060 |
| 8 | **Code Walkthrough** — Every key function in `lloyd_max.py`, `turboquant.py`, and `compressors.py` explained |
| 9 | **Researcher's Map** — Implications for vector search, open questions, 5 papers to read, 3 things to try |

The tutorial was produced by a multi-agent pipeline (researcher → writer → editor → assembler) using Claude Code. See [`tutorial/agent_team_documentation.md`](tutorial/agent_team_documentation.md) for the full account of how the team worked.

### 📚 [`tutorial/agent_team_documentation.md`](tutorial/agent_team_documentation.md)

A detailed behind-the-scenes account of the AI agent team that produced the tutorial — including agent profiles, the task system, communication protocol, turn-by-turn narrative, and quality metrics from the write-edit loop.

---

## Background

When an LLM generates text, it stores a **key** and **value** vector for every token it has seen, in every layer. This is the KV cache — the model's working memory. At 8K tokens on a 36-layer model like Qwen2.5-3B, this cache is **289 MB** in FP16. On a 12GB GPU, the KV cache — not the model weights — becomes the bottleneck for long context.

TurboQuant compresses this cache by quantizing the vectors to 2-4 bits per coordinate, achieving 3-7x compression with minimal impact on attention accuracy.

---

## How TurboQuant Works

The algorithm has two stages:

### Stage 1: Random Rotation + Lloyd-Max Quantization

Each vector is multiplied by a random orthogonal matrix (generated via QR decomposition of a Gaussian matrix). This rotation is the key trick — it makes every coordinate of the resulting vector follow a predictable bell-curve distribution, well-approximated by N(0, 1/d) for typical head dimensions.

Because the distribution is known and coordinates become nearly independent, we can design an **optimal scalar quantizer** (Lloyd-Max) for each coordinate independently. The Lloyd-Max algorithm finds the best set of "buckets" to round values into, minimizing mean squared error.

### Stage 2: QJL Residual Correction (1 bit)

The MSE-optimal quantizer from Stage 1 introduces a small bias in dot products. Since attention scores are dot products between queries and keys, this bias accumulates and skews model outputs.

The Quantized Johnson-Lindenstrauss (QJL) transform fixes this. It takes the quantization residual (the error from Stage 1), projects it through a random Gaussian matrix S, and stores just the **sign** (+1 or -1) of each projection — exactly 1 bit per dimension. This single bit makes the inner product estimate **mathematically unbiased**.

The combined estimator for `<query, key>` is:

```
<q, k> ≈ <q, k_mse> + ||residual|| * sqrt(pi/2) / m * <S @ q, sign(S @ residual)>
```

Where `S` is the random projection matrix, `k_mse` is the Stage 1 reconstruction, and `residual = k - k_mse`.

### Why It Works Despite High Per-Vector Error

An important subtlety: the per-vector reconstruction error is significant (23-44% relative error depending on bit-width). If you decompress the vectors and feed them to standard attention, the model produces garbage. But TurboQuant doesn't need accurate vector reconstruction — it needs accurate **inner products** (attention scores). The QJL correction ensures these are unbiased with variance O(1/d). The attention distribution over tokens is preserved even when individual vectors look quite different from the originals.

---

## Experimental Results

### Synthetic Vector Tests

| Bits | Bias (inner product) | Correlation with true IP |
|------|---------------------|------------------------|
| 2-bit | +0.001 | 0.80 |
| 3-bit | +0.000 | 0.93 |
| 4-bit | +0.000 | 0.98 |

Near-zero bias at all bit-widths. Needle-in-haystack: 9/9 exact retrieval across all bit-widths and sequence lengths (512, 2048, 8192).

### Real Model Validation — Qwen2.5-3B on RTX 3060

**Compression:**

| Config | KV Cache (8K ctx) | Compression |
|--------|------------------|-------------|
| FP16 baseline | 289 MB | 1.0x |
| TurboQuant 4-bit | 76 MB | **3.8x** |
| TurboQuant 3-bit | 58 MB | **5.0x** |
| TurboQuant 2-bit | 40 MB | **7.3x** |

**Attention fidelity at 3-bit (averaged across 36 layers):**

| Context | Cosine Similarity | Top-1 Match | Top-5 Match |
|---------|------------------|-------------|-------------|
| 2K | 0.9961 | 85% | 94% |
| 4K | 0.9955 | 75% | 88% |
| 8K | 0.9945 | 86% | 94% |

**3-bit is the practical sweet spot:** 5x compression with 99.5% attention fidelity. On a 12GB GPU, this is the difference between fitting 8K and 40K context with the same model.

---

## Quick Start

```bash
pip install -r requirements.txt

# Synthetic algorithm validation (no GPU required)
python -m turboquant.test_turboquant

# Real model validation (requires CUDA GPU, ~6GB VRAM)
python -m turboquant.validate
```

---

## Project Structure

```
turboquant-pytorch/
├── __init__.py              # Package exports
├── lloyd_max.py             # Lloyd-Max optimal scalar quantizer
├── turboquant.py            # Core: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
├── compressors.py           # Asymmetric inner product compressors (V2)
├── test_turboquant.py       # Synthetic algorithm tests
├── validate.py              # Real model attention comparison (Qwen2.5-3B)
├── requirements.txt
│
├── tutorial/
│   ├── turboquant_tutorial.md       # ← The deep tutorial (~9,500 words, 9 sections)
│   ├── agent_team_documentation.md  # ← How the agent team produced the tutorial
│   ├── research_notes.md            # Technical research notes (used to write tutorial)
│   ├── final_section_1.md           # Individual approved section files
│   ├── final_section_2.md
│   ├── ...
│   └── final_section_9.md
│
└── docs/
    ├── turboquant_blog_google.txt   # Google Research blog post
    ├── transcript.txt               # YouTube validation experiment transcript
    └── README.txt                   # Links to papers and original repo
```

---

## Acknowledgements

This repository is a fork of **[tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)** by Tonbi Studio ([@tonbistudio](https://x.com/tonbistudio)). All implementation code, experimental design, and validation results originate from that project. This fork adds educational materials only.

### Primary Sources

**1. YouTube — Validation Experiment**
> *"Testing Google's TurboQuant Approach: I Got 5x Compression with 99.5% Accuracy!"*
> Tonbi Studio · [https://www.youtube.com/watch?v=iD29muStx1U](https://www.youtube.com/watch?v=iD29muStx1U)
>
> The original validation experiment: implementing TurboQuant from scratch in Claude Code, testing on Qwen 2.5 3B on an RTX 3060, and verifying the compression and attention fidelity claims. The tutorial in this repo is directly inspired by and based on this video and the underlying codebase.

**2. TurboQuant Paper (ICLR 2026)**
> Amir Zandieh, Praneeth Kacham, Insu Han, Majid Daliri, Lars Gottesbüren, Rajesh Jayaram, Vahab Mirrokni
> *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
> [https://arxiv.org/pdf/2504.19874](https://arxiv.org/pdf/2504.19874)
>
> The primary paper. Introduces the two-stage quantization framework (PolarQuant + QJL), proves near-optimal distortion rates, and demonstrates zero-accuracy-loss KV cache compression at 3 bits.

**3. PolarQuant Paper (AISTATS 2026)**
> *"PolarQuant: Quantizing KV Caches with Polar Transformation"*
> [https://arxiv.org/pdf/2502.02617](https://arxiv.org/pdf/2502.02617)
>
> The Stage 1 component paper. Introduces the polar coordinate reparameterization that eliminates per-block memory overhead in classical vector quantization.

**4. Google Research Blog**
> Amir Zandieh and Vahab Mirrokni
> *"TurboQuant: Redefining AI efficiency with extreme compression"*
> Google Research · March 24, 2026
> [https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
>
> Accessible overview of TurboQuant, QJL, and PolarQuant with benchmark results and implications for KV cache and vector search.

### Additional References

- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482) — The QJL technique underlying Stage 2
- [QJL Reference Implementation](https://github.com/amirzandieh/QJL) — Original CUDA implementation by the QJL authors
- [PolarQuant Reference Implementation](https://github.com/ericshwu/PolarQuant)
- Credit to [@Prince_Canuma](https://x.com/Prince_Canuma) for the Mac implementation that inspired the original validation experiment

---

## License

MIT
