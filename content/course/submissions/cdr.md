---
title: Design Review — Demo-Aware Offline Policy Learning on RoboTurk Bread Bin Demonstrations
author: Soorej Nair
date: 2026-04-16
---

# Critical Design Review: Demo-Aware Offline Policy Learning on RoboTurk Bread Bin Demonstrations

## 1. Project Summary

This project studies offline policy learning for robotic manipulation using
demonstration data drawn from the RoboTurk bread-bin task. The central goal is
to learn action prediction policies from recorded state-action trajectories and
to evaluate how well those policies reproduce expert behavior on held-out
demonstrations. In particular, the codebase implements two sequence-conditioned
policy models. The first is a Gaussian autoregressive-style policy that
predicts the next action from a window of preceding states. The second is a
diffusion-based denoising policy that reconstructs actions through an iterative
reverse process conditioned on the same state history.

The system is designed around a principled offline learning pipeline. Raw HDF5
demonstrations are transformed into a cached intermediate representation that
preserves demonstration boundaries, step counts, and action construction rules.
Thereafter, the data are partitioned at the demonstration level, normalized
using statistics computed only from the training split, and consumed through a
windowed sequence dataset that exposes short temporal context to the policy.
Training, evaluation, checkpointing, and result logging are all integrated into
one reproducible workflow.

From a design standpoint, the project is focused and well scoped. It does not
attempt to solve the broader problem of closed-loop robotic deployment.
Instead, the focus is on a smaller but still important question. It looks at whether a sequence conditioned offline learner can capture expert action trajectories well enough to generalize to new demonstrations from the same task family. 

## 2. System Architecture

### Demonstration ingestion
The system begins with RoboTurk HDF5 files that contain state trajectories and
control signals for individual demonstrations. Each demonstration provides a
sequence of 73-dimensional states together with motion and gripper commands.
The preprocessing stage reads these demonstrations directly from disk and
constructs an 8-dimensional action vector from translational deltas,
quaternion deltas, and the gripper actuation signal.

### Cached dataset representation
Rather than re-reading and flattening the source file for every experiment, the
pipeline writes a cached NPZ file whose name is derived from a configuration
dependent cache key. This artifact stores the full state and action arrays
along with demonstration offsets, lengths, and names. Consequently, the cache
preserves the compactness of flat arrays while retaining the structural
information needed for trajectory-aware training and evaluation. This is an
important design choice because it reduces preprocessing cost without erasing
the unit of generalization, which in this setting is the demonstration rather
than the individual timestep.

### Demo selection and partitioning
The preprocessing and split logic support deterministic selection of
demonstrations through a random seed. In addition, the system supports three
selection modes, namely first-in-order, random sampling, and stratified
sampling by demonstration length. After selection, the dataset is partitioned
into train, validation, and test subsets at the demonstration level. As a
result, every demonstration appears in exactly one split. This separation is
essential because it prevents information leakage through adjacent timesteps
from the same recorded episode.

### Sequence dataset
Training samples are built from fixed-length windows of past states. Each
window is left padded where necessary, and a corresponding padding mask is
provided to the model. The target is the action taken at the last valid step in
that window. This representation is computationally efficient, yet it still
exposes short-term temporal structure. Accordingly, the policy is trained to
map recent state history to the expert action that follows.

### Policy models
The autoregressive-style policy projects each state in the window to a latent
space, processes the sequence through a Transformer encoder, and predicts the
mean and log standard deviation of a diagonal Gaussian action distribution. The
diffusion policy uses a related state encoder, but it combines the encoded
history with a noisy action sample and a timestep embedding. It then predicts
the noise component that must be removed during the reverse diffusion process.

## 3. Input and Output Representation

### Inputs

| Tensor or artifact | Shape | Meaning |
|---|---|---|
| `states` | $(N, 73)$ | Flattened state vectors recovered from the selected demonstrations |
| `actions` | $(N, 8)$ | Concatenated end-effector translation, orientation delta, and gripper command |
| state window | $(W, 73)$ | Context window of recent states, with default window length 16 |
| padding mask | $(W,)$ | Boolean mask that marks padded elements in shorter histories |
| normalization statistics | 73-D and 8-D vectors | Mean and standard deviation computed from training demonstrations only |

The state representation is taken directly from the RoboTurk demonstration
format. The action representation is compact and interpretable. It preserves
the structure of the original control signal while remaining small enough to be
modeled directly as a continuous vector. Because the policy operates on windows
of states rather than on isolated inputs, the representation also encodes short
temporal context. In practice, this is necessary for manipulation tasks in
which consecutive states may be locally ambiguous unless they are interpreted
within the trajectory that produced them.

### Outputs

| Model | Output | Interpretation |
|---|---|---|
| AR policy | $\mu \in \mathbb{R}^8$, $\log \sigma \in \mathbb{R}^8$ | Parameters of the predicted Gaussian action distribution |
| Diffusion policy | $\hat{\epsilon} \in \mathbb{R}^8$ | Predicted noise for reverse denoising at the sampled timestep |
| Evaluation output | $\hat{a}_{1:T} \in \mathbb{R}^{T \times 8}$ | Full predicted action trajectory for a held-out demonstration |

At evaluation time, both policies ultimately produce denormalized action
sequences. These trajectories can then be compared directly to the expert
actions stored in the dataset. This is a sensible design because it keeps the
training numerically stable while preserving interpretability in the reported
metrics.

## 4. Training Algorithm

### Preprocessing objective
The preprocessing stage constructs one action vector per timestep according to

$$
a_t = \left[\Delta p_t,\; \Delta q_t,\; g_t\right] \in \mathbb{R}^8
$$

and pairs it with the corresponding state vector $s_t \in \mathbb{R}^{73}$.
For each demonstration $d$ with length $T_d$, the system preserves the ordered
sequence

$$
\mathcal{D}_d = \{(s_t, a_t)\}_{t=1}^{T_d}.
$$

The full offline corpus is therefore a collection of demonstrations rather than
a bag of independent state-action rows. This distinction is conceptually
important because the learning problem is framed around policy behavior on
trajectories.

### State normalization
Normalization statistics are computed only from the training split. Let
$\mu_s, \sigma_s$ denote the mean and standard deviation of the training states,
and let $\mu_a, \sigma_a$ denote the corresponding action statistics. Then the
training representation is

$$
\tilde{s}_t = \frac{s_t - \mu_s}{\sigma_s},
\qquad
\tilde{a}_t = \frac{a_t - \mu_a}{\sigma_a}.
$$

This design is methodologically sound because it prevents leakage from the
validation and test distributions into the normalization pipeline.

### Autoregressive-style policy objective
Given a state window $x_t = (\tilde{s}_{t-W+1}, \ldots, \tilde{s}_t)$, the AR
policy predicts Gaussian parameters $(\mu_t, \log \sigma_t)$ and optimizes the
negative log-likelihood

$$
\mathcal{L}_{\mathrm{AR}}
= - \frac{1}{B} \sum_{i=1}^{B}
\log \mathcal{N} \bigl(\tilde{a}_i \mid \mu_i, \sigma_i^2 I \bigr).
$$

This loss is appropriate because the action space is continuous and because the
model is meant to capture both central tendency and uncertainty. Although the
distribution is diagonal, the Transformer encoder can still represent temporal
dependencies across the state history before the action head factorizes the
output dimensions.

### Diffusion policy objective
For the diffusion policy, the system samples a diffusion timestep
$t \sim \mathrm{Uniform}\{0, \dots, T-1\}$ and draws Gaussian noise
$\epsilon \sim \mathcal{N}(0, I)$. Using a cumulative noise schedule
$\bar{\alpha}_t$, it constructs a noisy action

$$
\tilde{a}^{(t)} = \sqrt{\bar{\alpha}_t} \, \tilde{a}
+ \sqrt{1 - \bar{\alpha}_t} \, \epsilon.
$$

The network is trained to predict $\epsilon$ from the state window, the noisy
action, and the timestep embedding. The optimization target is

$$
\mathcal{L}_{\mathrm{diff}}
= \frac{1}{B} \sum_{i=1}^{B}
\left\lVert \epsilon_i - \hat{\epsilon}_i \right\rVert_2^2.
$$

At inference time, the policy begins with Gaussian noise and applies iterative
denoising steps to recover an action estimate. This design follows the standard
logic of diffusion modeling and is appropriate for continuous action synthesis.

### Model selection
Training proceeds over multiple epochs, and validation is performed at a fixed
interval. The best checkpoint is selected using the mean trajectory RMSE on the
validation set. This is a reasonable criterion because the core purpose of the
system is to model complete demonstrations rather than isolated timesteps.

## 5. Data Strategy

### Dataset scope

| Source | Scope | Notes |
|---|---|---|
| RoboTurk `bins-Bread` HDF5 | 1069 demonstrations | Primary data source for the present study |
| Default experiment subset | 128 demonstrations | Seed-controlled and optionally length stratified |
| Validation smoke run | 16 demonstrations | Used for execution verification and pipeline testing |

The data strategy is deliberately narrow. The project focuses on a single task
family rather than on broad multi-task generalization. For the purposes of a
design study, this is acceptable because it allows the system to isolate
questions of modeling, representation, and evaluation without confounding them
with large task heterogeneity. Nevertheless, the design remains extensible. The
cache builder and split logic do not assume any specific number of
demonstrations, and the experiment pipeline can be re-used on larger subsets or
alternative RoboTurk tasks with minimal modification.

### Strengths of the data design

- Demonstrations remain intact throughout preprocessing, splitting, and evaluation.
- Selection is deterministic under a seed, which supports reproducibility.
- Length stratification reduces the risk of pathological splits dominated by very short or very long trajectories.
- Train-only normalization preserves methodological correctness.
- Cached preprocessing avoids repeated HDF5 parsing and reduces experiment overhead.

### Limitations of the data design

- The project still evaluates only one manipulation task by default.
- Demonstration length is the only stratification variable currently supported.
- No semantic metadata are used to group trajectories by difficulty, failure mode, or behavioral style.
- The evaluation setting replays recorded states rather than simulating the state transitions induced by predicted actions.

These limitations do not invalidate the study. However, they do bound the
strength of the conclusions that can be drawn. In particular, the present
system can support claims about held-out demonstration modeling within one task
distribution. It cannot yet support strong claims about transfer, robustness,
or interactive control.

## 6. Evaluation Plan

### Metrics

| Metric | Definition |
|---|---|
| `step_mse` | Mean squared error per scalar action element |
| `step_rmse` | Root mean squared error per scalar action element |
| `step_mae` | Mean absolute error per scalar action element |
| `step_max_abs_error` | Maximum absolute scalar action error |
| `step_cosine_similarity` | Mean cosine similarity between predicted and expert action vectors |
| `step_gripper_sign_accuracy` | Accuracy of the predicted gripper command sign |
| `step_mean_l2` | Mean Euclidean distance between predicted and expert actions |
| `step_success_at_0.05` | Fraction of timesteps with action L2 error below 0.05 |
| `step_success_at_0.10` | Fraction of timesteps with action L2 error below 0.10 |
| `trajectory_rmse_mean` | Mean trajectory RMSE over held-out demonstrations |
| `trajectory_rmse_std` | Standard deviation of trajectory RMSE |
| `trajectory_rmse_median` | Median trajectory RMSE |
| `trajectory_final_l2_mean` | Mean final-step action error across demonstrations |
| `trajectory_nll_mean` | Mean action negative log-likelihood for the AR policy |

This metric suite is well chosen because it covers several distinct failure
modes. Pointwise error statistics capture average fidelity. Cosine similarity
captures directional agreement. Threshold-based measures indicate how often the
model remains within a tolerable local error bound. Finally, trajectory level statistics help show if some demonstrations are consistently more difficult than others. When you look at all of these together, you get a clearer understanding of the model’s behavior instead of relying on just one overall loss value.

### Evaluation protocol
The evaluation protocol is based on trajectory replay. For each held-out
demonstration, the model is given the recorded state history and asked to
predict the expert action at every timestep. The resulting predicted action
trajectory is then compared to the reference trajectory stored in the dataset.
Prediction files can also be written to CSV for per-step inspection.

This protocol is appropriate for offline learning because it isolates action
modeling quality without introducing confounds from imperfect environment
simulation. Even so, it does not measure compounding state error under closed
loop deployment. Therefore, the evaluation should be understood as a strong
offline proxy rather than as a substitute for embodied rollout testing.

### Observed smoke-test behavior
The codebase was executed successfully in a one-epoch smoke test using 16
demonstrations. The resulting run directory was
`logs/smoke_test_v3`. Under this setting, the autoregressive-style policy
achieved lower trajectory RMSE than the diffusion policy. This finding is
useful as a verification signal because it shows that the pipeline can
differentiate model behavior and persist all outputs correctly. However, the
sample is too small to make a final conclusion on the model performance, and a long run is still necessary.

## 7. Implementation Status

**Implemented and working:**

- Cached preprocessing with configuration-dependent cache keys
- Demonstration-level subsampling and partitioning
- Train-only state and action normalization
- Windowed sequence dataset with padding masks
- Transformer-based Gaussian policy
- Transformer-conditioned diffusion denoising policy
- Validation-based checkpoint selection
- Persistent experiment logs, JSON summaries, and prediction exports
- Demo replay mode using saved checkpoints

**Not yet fully realized:**

- Closed-loop environment rollout evaluation
- Automated unit and integration tests
- Multi-task training and testing across additional RoboTurk tasks
- Richer stratification beyond demonstration length
- Visualization tools for trajectory-level diagnostics

**Critical-path concern.** The principal limitation is the absence of
closed-loop evaluation. Without a simulator or environment wrapper, the system
cannot yet determine how prediction errors alter future states during actual
execution. As a consequence, the present pipeline should be interpreted as a
trajectory modeling benchmark rather than as a deployment-ready robotics stack.

## 8. Future Work

- Add a closed-loop evaluation interface in simulation so that predicted actions affect future states
- Extend the study to additional RoboTurk tasks and compare single-task and multi-task training
- Introduce richer split criteria based on trajectory difficulty, outcome quality, or behavioral regime
- Add diagnostic visualizations for per-dimension action error and temporal drift across trajectories
- Evaluate sensitivity to context window size, diffusion schedule design, and model depth
- Run repeated experiments across multiple random seeds and report confidence intervals
- Add a formal automated test suite for preprocessing, caching, splitting, checkpoint loading, and evaluation
