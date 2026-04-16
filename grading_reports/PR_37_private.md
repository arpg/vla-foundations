# PRIVATE GRADING REPORT - PR #37

**Student:** cKohl10
**Branch:** scratch-1-ckohl10
**Graded:** 2026-02-16 20:16:28

---

## AUTOMATED SCORES (67/70 pts)

| Component | Score | Max | Details |
|-----------|-------|-----|---------|
| Causal Attention | 15 | 15 | ✅ Passed |
| RMSNorm | 10 | 10 | ✅ Passed |
| Training | 10 | 10 | ✅ Passed |
| RoPE | 15 | 15 | ✅ Passed |
| Model Architecture | 10 | 10 | ✅ Passed |
| Code Quality | 7 | 10 | ❌ Failed |

---

## MANUAL REVIEW REQUIRED

### Documentation (0-30 pts): _____ / 30

Report files found:
- content/course/submissions/scratch-1/cKohl10.mdx
- README.md

Check for:
- [ ] Loss curve visualization (clear, labeled)
- [ ] Attention map visualization (interpretable)
- [ ] The Audit: causal mask removal analysis

### Mastery Components (0-10 pts): _____ / 10

Features detected:
- RoPE vs Sinusoidal ablation study

---

## FINAL SCORE

**Automated:** 67/70
**Documentation:** _____ / 30
**Mastery:** _____ / 10
**Adjustments:** _____ (e.g., -1 for programmatic fixes like missing dependencies)

**TOTAL:** _____ / 100

---

## TEST DETAILS


### Code Quality

- **test_import_success**: ✅ PASS (4/4 pts)
  - ✅ Code imports successfully.
- **test_no_syntax_errors**: ✅ PASS (3/3 pts)
  - ✅ Test passed.
- **test_no_todos_left**: ❌ FAIL (0/3 pts)
  - ❌ Test failed.

### Rmsnorm

- **test_rmsnorm_implementation**: ✅ PASS (5/5 pts)
  - ✅ RMSNorm implemented correctly with proper normalization and learnable scale.
- **test_rmsnorm_numerical_stability**: ✅ PASS (5/5 pts)
  - ✅ Test passed.

### Causal Attention

- **test_causal_mask_leakage**: ✅ PASS (8/8 pts)
  - ✅ Perfect! Your causal mask correctly prevents future token leakage.
- **test_causal_attention_shape_preservation**: ✅ PASS (7/7 pts)
  - ✅ Test passed.

### Rope

- **test_rope_embeddings**: ✅ PASS (15/15 pts)
  - ✅ RoPE correctly applied to Q and K tensors.

### Training

- **test_training_convergence**: ✅ PASS (10/10 pts)
  - ✅ Excellent! Your model trains successfully and loss converges.

### Model

- **test_model_forward_pass**: ✅ PASS (5/5 pts)
  - ✅ Model forward pass works end-to-end with correct output shapes.
- **test_model_has_trainable_parameters**: ✅ PASS (5/5 pts)
  - ✅ Model has the expected number of trainable parameters.
