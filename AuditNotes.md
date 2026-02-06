# Group D: The Planning and Safety Lab

## Aritra Project Discussion

### Pitch / Initial Dissolve
**Question:** Are token-level uncertainty signals (entropy, perplexity) from a VLA’s action distribution *sufficient* indicators that the agent is near an **irreversible/critical state**?

**Technical delta:** Provide an inference-time “near-failure” indicator for VLAs without retraining the model.

---

### Constraints / Scope
- No large-scale training; primarily **inference-time analysis**.
- Target: **irreversibility in state space** (unsafe/absorbing states), not just “bad actions.”
- Keep the core demo simple and measurable.

---

### Test Environments
1. **Primary:** visual MDP (gridworld / simple visual control) with explicitly defined irreversible states.
2. **Stretch:** robotic arm simulator with safety/irreversibility conditions (collisions, constraint violations, unrecoverable configs).

---

### Approach
Generate action distributions from the VLA and test whether uncertainty predicts future failure:
- **Next-action only:** entropy/perplexity at state \(s_t\).
- **Limited lookahead tree (depth d):** sample actions + simulate transitions; aggregate uncertainty.
- **Monte Carlo rollouts:** sample trajectories to termination; label whether they end in irreversible states; test early-warning value.

---

### Signals / Metrics
- Action distribution **entropy**
- Action/token **perplexity** (or NLL of selected action)
- (Optional) disagreement/variance across sampled action sequences

---

### Extension (if signals are weak)
Learn an explicit **safe/unsafe boundary** from rollouts (e.g., constraint model / CBF-style classifier) using VLA-generated trajectories.

---

### Load-Bearing Wall / Risks
- **Inference cost** (many samples/rollouts per state)
- **Weak signal**: entropy/perplexity may not correlate with proximity to irreversible states (confidently wrong / noisy)




## Zakariya Project Discussion

### Pitch / Initial Dissolve:
The core robotic bottleneck is Autoregressive Error Accumulation. Vision-Language-Action (VLA) models typically operate in an open-loop, token-by-token manner. In this paradigm, a single quantization error in an action token leads to compounding trajectory drift that the model cannot recover from without explicit long-horizon reasoning. This proposed "delta" is the correct architectural choice because it transitions the VLA from a deterministic controller to a Stochastic Reference Policy ($\overline{\pi}$) within a formal planning framework. We know this is the problem because state-of-the-art VLA deployments fail in long-horizon tasks where error recovery and future consequences are not explicitly modeled.

### Technical Delta:

The architecture fuses two distinct advancements to address current VLA limitations: 

- **VLA-as-Reference ($\overline{\pi}$):** The VLA provides the prior distribution for action sampling, allowing the solver to maximize rewards while minimizing KL-divergence from the VLA’s "policy".

- **VOI-Guided Branching:** Using a meta-level decision space (the VOI-POMDP), the planner selectively disregards observations when the Value of Information (VOI)—the expected performance gain from reasoning about observations—is low.

### Information Decay:

Although this approach extends the reasoning horizon beyond pure VLA action execution, it can still lose information through **adherence constraints** and **selective observation processing**:

- **Reference Adherence (KL Bottleneck):** In the reference-based POMDP objective, the KL-divergence penalty includes a tuning parameter that directly controls how much the planner is allowed to deviate from the base VLA policy $\overline{\pi}$. If this penalty is set too high, the solver will remain overly coupled to $\overline{\pi}$, suppressing corrective deviations even when the belief state indicates compounding error or looming failure. If set too low, the planner may discard useful inductive bias from $\overline{\pi}$ and waste queries exploring implausible action sequences.

- **Observation Gating (VOI Bottleneck):** VOI-MCP introduces a criterion parameter $\kappa$ that governs when the planner branches on observations. This parameter must be tuned carefully: if $\kappa$ is too aggressive, the planner may **ignore critical observations** (e.g., obstacle proximity, contact events, slip) that are necessary for task success and safe execution. If $\kappa$ is too conservative, the search reverts toward exhaustive action-observation branching, reintroducing the computational dilution that makes long-horizon reasoning intractable.

### Physicality Gap

This approach is still subject to the physicality gaps of using a VLA-only approach, including dynamics mismatch from learning on noisy data, action tokenization resolution etc. Additionally, any assumptions on the world model used in online tree search are prone to clash with real-world observations and transitions.

## Jimmy Project Discussion

### Pitch / Initial Dissolve:
The core idea is safety alignment of VLAs and VLMs - particularly, training VLAs that can work with only partially observable states (high latency sensor data) by having some sort of understanding of safety hazards in scene that are both currently present, or might in the future be present. Simple case example would be robot in dynamic environments (dynamic in the sense of moving objects, or scenes that are temporally changing ~ like a fire that is developing, or an object that could move within some semantic understanding of how it moves). The approach would be to have the model take a snapshot, query a VLM to create a high level topological map of features in the scene (like object detection), then using a diffusion model to roll out trajectories for objects or features in the scene either via some estimator/controller type of framework, or purely just safety bounding based on semantic cues. E.g. take a snapshot, query the VLM for scene understanding - including some information about each detected feature on where it could be heading (output could be like object A looks like it is moving right at a slow rate, then we send that information to some diffusion model to project forward in time). In low latency maybe we can just continually take the straight path until the occlusion forces you to move around it, this approach hopes to get an agent (the robot) to pre-emptively move around the region that could be in the future occluded. Plan is to assume worst case, even if it means having to stop to reassess. Outputs would be simple and agnostic like move right, left, forwards, diagonal, etc. with some magnitude, which can be converted to robot motor commands.

### Training
Still not a fully formed idea, but would need to have some internet priors to understand what exactly constitutes as a hazard in the specific environment of choice. After that, most training would need some sort of labelled dataset for estimate of trajectory of features or development of features in scene...this is likely a problem not widely solved yet and not sure if datasets for this exist. Object detection is already solved, but need this extra layer to feed into a diffusion model.

### Technical Delta:
Safety alignment of VLAs with understanding of hazards in scene, as well as a diffusion-based rollout of scene development (trajectories) to add bounding overhead for robot navigation. Aimed towards more resource constained platforms with high latency that can not continually resense, and need to have a good enough semantic grasp of worst case scenarios so we can inflate safety bounds for planning.

### Load-Bearing Walls and Failure Points:
1. Data training - where exactly can I find data for this sort of training
2. Scene understanding - generalization can only be as good as understanding what sort of hazards are present in scene type. Could also just stick to one type of environment and tailor narrative towards that (like human-dense environment, etc.)
3. A lot of the inputs and outputs are abstractions and not very discrete, how does this ensure safety or generalization to safety? Will have sensors more than just RGB so we can do physical alignment, but trajectories for example are hard to scale for prediction.

## Zack Project Discussion

### Pitch / Initial Dissolve:
Behavior Tree–based autonomy collapses rich execution outcomes into a binary success/failure signal, discarding the cause of failure (occlusion, degraded sensing, millimeter-scale pose error, contact/torque anomalies). This information decay occurs precisely where structured autonomy meets the physical world, making minor, recoverable failures indistinguishable from logical task failure and resulting in brittle retries or aborts under partial observability. This is the correct bottleneck to solve because the missing information already exists in the system but is not preserved or reasoned over.

### Technical Delta:
I introduce a failure-triggered Vision–Language–Action overviewer that activates only when a BT action fails. The overviewer consumes raw visual observations and physics-preserving execution context to semantically interpret the failure and propose a constrained recovery action, which is compiled into a temporary parameterized BT subtree. This restores semantic information at the failure boundary while preserving the safety, interpretability, and certifiability of Behavior Trees.

### Information Decay
Information is lost when continuous physical signals (contact state, torque trends, expected vs. observed motion) are projected into coarse symbolic failure tokens. If these signals are not preserved—along with minimal temporal context—the model cannot reason about physical causality and is dead on arrival. The proposed system explicitly encodes physics-relevant context and limits Internet priors to semantic interpretation rather than control.

### Physicality Gap
In messy, high-dynamic environments, the first failure mode is Internet priors overpowering physical reality under degraded sensing (occlusion, dust, blur). Without strong physical grounding, visually plausible but physically impossible recoveries are proposed. This is mitigated by freezing or low-rank tuning the vision backbone, enforcing BT safety guards, and grounding recovery decisions in embodied failure outcomes.

### Load-Bearing Walls and Failure Points:
1. Failure modes must map to distinct, actionable recoveries.
2. Recovery primitives must be expressive yet safety-constrained.
3. Execution context must preserve physical and temporal information.
4. Physics must dominate semantics when priors and reality disagree.


## Himanshu Gupta Project Discussion 
**(Vision + Particle Filter Belief Encoding → Action Decoder)**
---
### Pitch / Initial Dissolve
**Question:**  
If we explicitly provide a VLA with a belief state represented as a particle set, can it learn a belief-aware policy that chooses actions for both task progress and information gathering, without an explicit online tree search?
**Technical delta:**  
Build a VLA-style policy whose “state” is not just an observation embedding, but:
- a visual embedding `z_t^vision`, and  
- a learned, permutation-invariant encoding of the particle filter set  
  `z_t^belief = SetTransformer({(x_t^i, w_t^i)})`,  
trained end-to-end with the action decoder.
---
### Core Hypothesis
If a VLA is given structured access to belief uncertainty (via a particle set encoding), it can learn policies that:
- move toward the goal when belief is confident, and  
- move toward informative observations when belief is uncertain,  

without explicit search or reward shaping at inference time.
---
### Constraints / Scope
**Learning targets:**
- PF-set encoder (SetTransformer / DeepSets)
- action decoder/policy head

**Assumptions:**
- Particle filter dynamics and observation model are fixed and known.
- Environment is small enough to generate large numbers of trajectories.
**Scope:**
- Target low-dimensional POMDPs / BeliefMDPs suitable for class-scale experiments.
---
### Test Environments
**Primary:**  
Toy visual POMDP (LightDark 3D) where information gathering is necessary due to:
- occlusions,
- ambiguous observations,
- unknown agent position.
---
### Approach
At each timestep `t`:
1. **Belief Update (Particle Filter)**  
   Maintain a belief over the latent state using a particle filter:
   - Particle set: `{(x_t^i, w_t^i)}_{i=1}^N`
   - `x_t^i` are particle states
   - `w_t^i` are particle weights
2. **Vision Encoding**  
   Encode the current observation image `o_t` using a vision encoder:
	o_t -> z_t^vision
3. **Belief Encoding (SetTransformer)**  
Encode the particle set using a permutation-invariant SetTransformer:
	z_t^belief = ST({(x_t^i, w_t^i)}_{i=1}^N)

4. **Fusion and Action Decoding**  
Fuse the vision and belief embeddings and decode an action distribution:
	a_t ~ pi_theta(a | z_t^vision, z_t^belief)

---
### Training Options
- **Imitation Learning**  
Supervise actions using an oracle policy (e.g., particle filter + limited lookahead, MCTS, or a hand-designed information-gain heuristic).
- **Offline RL**  
Learn the policy from logged rollouts without further environment interaction (higher setup cost and risk).
- **Hybrid**  
Initialize with imitation learning, then perform limited policy improvement via short-horizon online rollouts.
---
### Metrics / What “Success” Means
- **Task performance:** reward/success rate compared to baselines
- **Information gathering behavior:**  
Does the policy move to reduce belief uncertainty when needed?
- **Belief-conditioned action sensitivity:**
- Hold observation fixed, vary belief → does action change appropriately?
- Hold belief fixed, vary observation → does action change appropriately?
- **Ablations:**
- vision-only vs belief-only vs vision + belief
- DeepSets vs SetTransformer vs mean/covariance encoding
---
### Why This Might Work
A particle set explicitly captures: multimodality (multiple hypotheses), uncertainty (spread / entropy), and structured belief geometry. A SetTransformer can preserve these properties more effectively than collapsing the belief to a mean or covariance.
---
### Load-Bearing Walls / Risks
- **Oracle problem:**  
If the supervisor is a planner (current plan), the model may simply distill tree search behavior, making the contribution amortized planning rather than planning removal.
- **Belief encoder leakage:**  
If particles are too close to ground truth, the task becomes trivial; if too noisy, the policy may fail to learn.
- **Generalization:**  
The learned mapping may overfit to specific PF statistics (number of particles, resampling scheme, noise model).
- **Long-horizon planning:**  
Direct belief-to-action mapping may fail on problems requiring explicit multi-step contingency reasoning.
