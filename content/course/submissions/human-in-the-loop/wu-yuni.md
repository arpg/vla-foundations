# If No One Is in the Driver’s Seat: Designing Safe, Bounded, and Legible Failure Modes for Driverless Systems  
**Yuni Wu**

---

## 1. Scenario

While developing a Unitree Go1 robot for hallway navigation, I encountered a fundamental limitation: the onboard compute is insufficient for real-time high-level reasoning. As a result, the system relies on a remote vision-language model (VLM) for human-aware decision-making. This introduces both **latency** and **network dependency** into a safety-critical loop.

This design parallels large-scale driverless vehicle systems such as Zoox, where high-level reasoning may be partially offloaded or monitored remotely. This leads to a critical question:

> **What happens when a fully driverless system loses connection to its human operator?**

In such cases, the system must not only remain physically safe, but also behave in ways that are predictable and interpretable to surrounding humans.

---

## 2. Failing Safely: From Control Loss to Minimal Risk Condition

A naive solution to communication failure is to immediately stop. However, abrupt stopping may itself introduce risk. Instead, the system should transition into a **minimal risk condition**.

### 2.1 Controlled Degradation

Upon detecting communication loss, the vehicle should:

- Gradually decelerate (avoid abrupt braking)  
- Maintain its current lane (no sudden lane changes)  
- Activate hazard lights  
- Continue basic perception and control  

If safe conditions allow:  
- Perform a controlled pull-over maneuver  

Otherwise:  
- Come to a stable stop while remaining predictable to surrounding drivers  

> The goal is not immediate halting, but **predictable and bounded behavior under failure**.

---

### 2.2 Minimal Risk Across System Scales

It is important to note that the definition of a “minimal risk condition” depends on the physical and social scale of the system.

For example:
- In autonomous driving, coming to a controlled stop within a lane is often considered a standard safe fallback behavior.
- In contrast, for a mobile robot operating on sidewalks or indoor corridors, remaining completely stationary may obstruct human flow and create secondary risks.

> Minimal risk is therefore not a universal state, but a context-dependent behavior shaped by environment and interaction dynamics.

---

### 2.3 Post-Stop Behavior

After stopping, the system should not power off. Instead, it should:

- Remain in a **monitored standby state**  
- Continue perception and environment awareness  
- Maintain system logging  
- Await reconnection or external intervention  

This ensures continued situational awareness and preserves accountability.

---

## 3. Autonomy Boundary Under Disconnection

Human-in-the-loop systems often assume continuous human availability. In practice, latency and communication failure invalidate this assumption. Therefore, autonomy must be **explicitly reduced under disconnection**.

### 3.1 Allowed Actions (Bounded Autonomy)

- Lane keeping  
- Controlled deceleration  
- Basic obstacle avoidance  
- Maintaining conservative but socially consistent following distance  

These actions are safe because their outcomes are predictable and bounded.

---

### 3.2 Disallowed Actions

In disconnected mode, the system must avoid actions that rely on implicit human coordination or negotiation.

This includes:

- Lane changes  
- Unprotected turns  
- Complex pedestrian interactions  
- Navigation through dense, dynamic environments  

More critically, the system should explicitly avoid entering scenarios such as:

- Unsignalized intersections  
- Left-turn waiting zones  
- Multi-agent merging situations  

These environments depend heavily on informal human communication, such as eye contact, timing negotiation, and shared driving norms.

> Without the ability to interpret or participate in these interactions, the system cannot safely operate in such contexts.

Therefore:

> The autonomy boundary must exclude environments where safety depends on implicit human cooperation rather than explicit rules.

---

### 3.3 Social Stability vs Physical Safety

Safety is not purely a physical property—it is also shaped by how other agents respond.

For example:
- If the vehicle maintains excessively large gaps, other drivers may repeatedly merge into its lane  
- This can destabilize traffic flow and induce unnecessary braking cycles  

Therefore:

> The vehicle should maintain a **conservative but socially consistent following distance**.

Additionally:

> Overly conservative behavior can itself become a source of risk by disrupting implicit coordination among human drivers.

Under disconnection, the system objective shifts:

> From efficiency → to predictability and stability.

---

### 3.4 Legal Constraints

Legal compliance becomes more critical when no human is present:

- The vehicle must strictly follow traffic laws  
- Responsibility attribution becomes more complex  
- Behavior must remain within clearly defined regulatory boundaries  

---

## 4. Trust, Communication, and the Risk of Transparency

When a system loses connection, the challenge becomes both technical and social.

### 4.1 The Legibility Problem

In human driving:
- Drivers communicate through motion and eye contact  
- Intent is implicitly understood  

In driverless systems:
- No visible driver exists  
- No direct explanation of behavior is available  

Thus:

> A system can be physically safe, yet socially perceived as unsafe.

---

### 4.2 The “Invisible Bomb” Problem

A disconnected vehicle may be perceived as unpredictable:

- Other drivers may over-avoid it  
- Traffic flow may become unstable  
- Individuals may attempt to test or exploit its behavior  

In extreme interpretations:

> The vehicle may be treated as an “unaccountable agent” within the environment.

This creates a paradox:

- Transparency may increase awareness  
- But may also increase fear, confusion, or opportunistic behavior  

---

### 4.3 Should the System Announce Disconnection?

A naive design might display messages such as:

- “Disconnected”  
- “No human in control”  

However, such explicit signaling may:

- Induce panic  
- Reduce perceived reliability  
- Encourage adversarial or exploitative behavior  

> Explicitly revealing internal failure states may reduce trust rather than improve safety.

---

### 4.4 Controlled Legibility

Instead of full transparency, systems should aim for:

> **Legibility without exposing vulnerability**

This includes:

- Standard hazard lights  
- Smooth, predictable motion  
- Regulation-compliant signaling  

The goal is:

- To make behavior understandable  
- Without exposing internal system failure  

---

## 5. Human Reconnection, Latency, and Control Instability

When connection is restored, the operator faces a different challenge:

- Delayed data  
- Unknown system transitions  
- High decision pressure  

---

### 5.1 Temporal Misalignment and Control Instability

Latency introduces a fundamental mismatch between perception and action.

For instance:
- The operator observes a frame at time *t − 500 ms*  
- A control command issued at that moment arrives at the vehicle at approximately *t + 500 ms*  

This creates a full one-second perception–action gap.

As a result:
- The operator is always reacting to outdated information  
- The system state may have already changed significantly  

This temporal misalignment is a primary cause of oscillatory or unstable control behavior.

> In such conditions, direct teleoperation may degrade system stability rather than improve safety.

---

### 5.2 Decision-Ready Summaries

Instead of raw sensor streams, the system should provide:

- Current vehicle state  
- Timeline of key events  
- Risk summary  
- Active constraints  

> The system should present **decision-ready information**, not raw data.

---

## 6. Accountability, Recording, and Policy Implications

When no human is actively controlling the system, accountability must be preserved through design.

### 6.1 Continuous Recording

Upon disconnection, the system should automatically:

- Activate full-scope sensor recording  
- Log system decisions and internal states  
- Preserve environmental context  

This protects:

- The vehicle (from false claims)  
- Other road users (from unsafe behavior)  
- System developers (for debugging and auditing)  

---

### 6.2 Legal and Policy Considerations

Technical solutions alone are insufficient.

Regulatory frameworks must define:

- Acceptable failure behaviors  
- Standardized signaling methods  
- Data recording requirements  
- Liability boundaries under disconnection  

---

## 7. Conclusion

Human-in-the-loop systems often assume that a human is always reachable. In real-world systems constrained by latency and network dependency, this assumption fails.

In driverless systems, loss of connection is not only a technical failure—it is a **social and systemic challenge**.

A robust system must:

- Degrade gracefully into a minimal risk condition  
- Restrict autonomy to bounded and predictable actions  
- Maintain controlled, interpretable behavior  
- Preserve accountability through continuous recording  

Ultimately:

> Safety is not only about avoiding accidents, but about maintaining trust in the absence of a human.

> Autonomy is not only limited by perception and control, but by the system’s ability to participate in human social conventions.