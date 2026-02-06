# Group D: The Planning and Safety Lab

## Aritra

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




## Zakariya

### Pitch: 
Inspired by partially observable markov decision processes how to assign value the information for long horizon planning. How to fine tune a VLA to quantify information value. “Emergent Temporal Abstractions inform Learing” train it on, train RL on it, instead of output of the tokens, they perform RL on the residual activation stream in each layer. Where does this application work best? You can discover this temporal abstraction works best in the middle layers.

### Questions:
Whats partially observable?
The state could be, 
What is this residual?
You get one on each layer, and train an RL model to get extended information about long horizon tasks. Helps ground actions over longer durations. Training over single output tokens losses the temporal information. Can we gather more information and control the residuals to prevent loss of information.
What is the value of receivng and excutuing outcomes? 
Sometimes these outcomes are completely unnecessary observation branching to make it easier

### Techincal Delta: (not discussed)


## Jimmy

### Pitch: 
Current research: Deployment of VLMs on constrained hardware. Attempting to add in the VLAs. Hoping to see high latency information like potential hazards. Using captures at a single time stamp to project trajectories for obects in the scene such as humans. Provide some understanding that this is a fire or walking around people in chairs to determine better trajectories. Diffusion models with static observer try to roll out the trajectory for 4 seconds and see what might happen.

### Questions:
Are you looking for the worst case?
Yes, for example moving in one direction with other people walking through, will the VLA be able to inform the system to stop or will it prevent collisions.
Tehcnical Delta: high liatency safety allignment of vlms vlas where the hazard is detected using diffusion based trajectory collision or object detection.
How will you identify this stuff? Diffusion models mostly based on what it is trained on. How do you align the VLA and diffusion model training?
Roll out diffusion policy and check for collision? Do you try multiple settings for the diffusion model? Planning on an abstraction and then diffuse out nitty gritty control. Good for non standard robotic platforms like insects.
At first thinking a block that can move in any direction

## Zack

### Pitch / Initial Dissolve:
Structured robotic autonomy systems based on Behavior Trees collapse rich execution outcomes into a binary success/failure signal. This creates an information bottleneck where the cause of failure—such as occlusion, degraded sensing, millimeter-scale pose error, or contact/torque anomalies—is lost before the policy can adapt. As a result, systems fail on minor, unforeseen conditions that should be recoverable, leading to brittle retry loops or aborts.

### Technical Delta:
I introduce a failure-triggered Vision–Language–Action overviewer that activates only when a BT action fails. The overviewer consumes raw visual observations and structured execution context to semantically interpret the failure and propose a constrained recovery action, which is compiled into a temporary parameterized BT subtree. This preserves the safety and interpretability of Behavior Trees while restoring the semantic information necessary to recover from unforeseen failures under partial observability.

### Load-Bearing Walls and Failure Points:
1. Can failure modes be meaningfully categorized in a way that links to distinct recoveries?
2. The recovery primitive set must be expressive enough to fix failures but small enough to be safe and learnable. Does this constraint prevent recovery success?
3. Information preservation in context encoding. Can you effectively convert the system state information into a useful context encoding?
4. Internet Priors Overpowering Physical reality. When physics (context embedding) and vision disagree, priors will confidently lie. 


## Himanshu Gupta  
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