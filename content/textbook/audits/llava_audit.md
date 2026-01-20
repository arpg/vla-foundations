---
title: "Technical Audit: LLaVA 1.5 Architecture"
author: "AI Thought Partner"
date: "2026-01-19"
---

# I. Formal Architecture

The LLaVA 1.5 architecture is a monolithic-encoder system that bridges vision and language via a non-linear projection.

### 1. State-Space and Latent Transitions
Let $X_v$ be the input image of resolution $336 \times 336$. The visual state is encoded into a grid of latent tokens:
$$Z_v = f_{\text{CLIP}}(X_v) \in \mathbb{R}^{N \times D_v}$$
where $N = 576$ patches and $D_v = 1024$. The transition to the LLM's latent space $H_l$ is governed by a two-layer MLP $g_\theta$:
$$H_v = W_2 \cdot \text{GeLU}(W_1 \cdot Z_v + b_1) + b_2$$
The joint probability for the autoregressive response $Y = \{y_1, \dots, y_L\}$ given instructions $X_q$ is:
$$P(Y \mid X_v, X_q) = \prod_{i=1}^{L} P(y_i \mid H_v, e_\psi(X_q), y_{<i})$$

### 2. Objective Function
The model is optimized using the standard Cross-Entropy loss over instruction-following data:
$$\mathcal{L}_{\text{LLaVA}} = -\sum_{i=1}^L \log P(y_i \mid H_v, X_q, y_{<i})$$

---

# II. The Instructor Audit Criteria

* **Dimensionality & Information Decay:** The primary bottleneck is the **ViT-L/14 patch size**. By compressing an image into $24 \times 24$ patches, sub-patch spatial geometry is discarded. Texture and fine-grained contact points are "blurred" into a single semantic vector.
* **Compute & Inference Reality:**
    * **Forward Pass:** ~210-250 GFLOPs.
    * **Edge Hardware (Orin):** Operates at approx **1.2 Hz**. Latency is dominated by the KV-cache during decoding.
* **The Semantic-Motor Gap:** LLaVA 1.5 is purely semantic. It identifies *what* an object is but fails to map that object to a continuous 3D action space because its tokens lack spatial-temporal depth.

---

# III. Failure Mode Analysis

1.  **Transparent Objects:** Suffers from "Refractive Hallucination," confusing backgrounds with the object itself.
2.  **Long-Horizon State Drift:** Fails to update internal world states, suggesting actions (e.g., "Pour") before prerequisites (e.g., "Open") are visually confirmed.