---
title: "Technical Audit: Prismatic VLM Architecture"
author: "AI Thought Partner"
date: "2026-01-19"
---

# I. Formal Architecture

Prismatic uses an **Ensemble Vision Backbone** to mitigate information decay found in monolithic systems.

### 1. Fused Latent Representation
Prismatic defines a fused visual state by concatenating features from a semantic encoder (SigLIP) and a geometric encoder (DINOv2):
$$Z_{\text{sem}} = f_{\text{SigLIP}}(X_v), \quad Z_{\text{geo}} = f_{\text{DINOv2}}(X_v)$$
$$Z_{\text{fused}} = [Z_{\text{sem}} \parallel Z_{\text{geo}}] \in \mathbb{R}^{N \times (D_{\text{sem}} + D_{\text{geo}})}$$
The projection to the LLM's embedding space is:
$$H_{\text{prism}} = g_\theta(Z_{\text{fused}}) = \text{MLP}(Z_{\text{fused}})$$

### 2. Objective Function
Prismatic utilizes a **Single-Stage Training** objective, focusing on direct preference or supervised fine-tuning:
$$\mathcal{L}_{\text{Prism}} = \mathcal{L}_{\text{SFT}}(H_{\text{prism}}, X_q, Y)$$

---

# II. The Instructor Audit Criteria

* **Dimensionality & Information Decay:** Reduces decay by including DINOv2, which retains **local pixel correspondences**. The bottleneck shifts to the **projection width**; if the MLP is too narrow, the geometric richness is sacrificed.
* **Compute & Inference Reality:**
    * **Forward Pass:** ~350-400 GFLOPs due to dual-encoder overhead.
    * **Edge Hardware (Orin):** Operates at **<1.0 Hz**. Requires aggressive INT4/FP8 quantization for real-time robotics.
* **The Semantic-Motor Gap:** Bridges the gap better via spatial features, but still outputs **textual tokens**. The gap remains an output interface problem.

---

# III. Failure Mode Analysis

1.  **Dynamic Velocity Aliasing:** Frame-based architecture lacks motion-aware states. High-velocity objects create "temporal blur," causing the model to target an object's past position.
2.  **Causal Reasoning Loops:** Prone to "Action Loops" where it repeats successful identification without executing the logic to move to the next sub-goal.