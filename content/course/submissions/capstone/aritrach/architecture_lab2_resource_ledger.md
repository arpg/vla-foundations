**AI Infrastructure Resource Ledger: The Reality Check**

*Thursday, 5\. March 2026*

**Engineer(s):** Aritra Chakrabarty

**Date:** 5 March 2026

**Project Title:** Token Entropy and Perplexity: Are they indicators of proximity to a sink-state?

**Phase 1: The Calendar (Working Backward)**

*If you are still writing your training loop on your Drop-Dead Date, your project will fail.*

1. Final Demo / Presentation Date: April 21, 2026 OR Apr 23, 2026  
2. Days required for Evaluation & Report: 14 days.  
3. Estimated Training Run Duration: 5 days. (NOT training, but inference)  
4. DROP-DEAD TRAINING START DATE: April 2, 2026  *(Subtract Line 2 & 3 from Line 1\)*

**Phase 2: The Stack & Compute Target**

1. Base architecture backbone, include encoders etc., add as many lines as you need: Qwen 3.5, and also Open VLA  
   Param Count: 9 Billion  
2. Target Compute Hardware:  
   \[x\] Local / Lab Node  
   \[ \] Cloud Rental (e.g., Lambda/AWS)  
3. Specific GPU(s): RTX 4090  
   Total VRAM Available: 24 GB

**Phase 3: The Memory Math (The VRAM Ledger)**

*Reference: 1B Parameters ≈ 2GB (Weights) \+ 2GB (Gradients) \+ 12GB (AdamW Optimizer States)*

1. Model Weights (BF16): 18 GB  
2. Gradients (BF16): 0 GB (*Inference*)  
3. Optimizer States: 0 GB (*Inference*)  
4. KV Cache & Activations Reserve: \~ 2-6 GB  
5. TOTAL REQUIRED VRAM: 20-24 GB

**The Strategy:** Is your Total Required VRAM \> Total Available VRAM?

\[ \] Yes \[x\] No

*If Yes, what is your exact mitigation strategy?* \[ \] FSDP / ZeRO-3 (Sharding across multiple GPUs)

\[ \] LoRA / QLoRA (Freezing base weights, avoiding optimizer state explosion)

\[ \] I am shrinking my model selection.

\[x\] Something else (explain): I may have to cut context size

**Phase 4: The Data Ingress (The I/O Bottleneck)**

1. Dataset Size: \<10 GB / \_\_\_\_\_\_\_\_\_\_ Total Samples (still selecting dataset to work with)

2. Format:  
   \[x\] Raw JPEGs/PNGs/trajectories/etc. in a folder (High I/O Risk)  
   \[ \] WebDataset / Sharded .tar  
   \[ \] Something else

3. Storage Location:  
   \[x\] Local NVMe SSD  
   \[ \] Network Drive  
   \[ \] S3/GCS Bucket  
   \[ \] Something else

*If using raw images or other large data quantities over a network or standard SSD, how will you prevent your dataloader from starving your GPUs?*

---

---

**Phase 5: Red Team Audit (Partner Review)**

**Reviewing Engineer:** Lorin Achey

*Look at your partner's ledger. Find the fatal flaw. Are they underestimating training time? Do they have a 40GB model fitting into 24GB of VRAM? Are they trying to load 2 million JPEGs sequentially?*

**The Fatal Flaw / Biggest Risk:**

* Resource contention on lab GPU resources could be a risk  
* Data storage is also a concern \- even if using HuggingFace for the dataset, we still need to pull that data over the network so that could be a limiting factor on the speed.

**Phase 6: The 72-Hour Contract**

*What exact, granular engineering task will you complete in the next 72 hours to unblock these risks? (e.g., "Write dummy dataloader script to verify throughput," "Run a 10-step overfitting batch to check VRAM usage.")*

**Deliverable by Monday:** 

I will choose:  
\- a dataset  
\- working environment  
\- appropriate action tokenization  
\- have script ready to run VLA locally and extract entropy \+ perplexity from action distribution  
\- explore if it’s possible to use VLA with a much simpler environment  
\- figure out the fastest way for SSD \-\> GPU data transfer for quick inference