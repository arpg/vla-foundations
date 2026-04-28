# When the Operator Goes Dark: Pre-Delegation as the Correct HITL Paradigm for L4 Autonomy

**Aritra Chakrabarty**

---

## 1. Introduction

The scenario is straightforward: an L4 vehicle is navigating an urban intersection under remote teleoperation when all redundant communication channels fail simultaneously. The vehicle must act. It cannot park in the middle of the intersection and wait for reconnection, nor can it request real-time approval from an operator who is unreachable. The conventional HITL model, where the system proposes and a human disposes, has no valid counterparty.

This document argues that for L4 remote teleoperation, HITL design must shift from *human approves actions in real time* to *human pre-delegates bounded policies at design time*. The operator's authority is not diminished; it is exercised earlier in the lifecycle, at a point where deliberation is possible and communication is guaranteed.

## 2. Failing Safely: The MRM as Safe Default

SAE J3018 defines the **Minimum Risk Maneuver (MRM)** as the procedural sequence an ADS-equipped vehicle executes to transition from its current state to a **Minimum Risk Condition (MRC)**, a stable, low-risk end state such as being stopped on the road shoulder with hazard lights active [SAE, 2020]. The distinction matters: MRM is the trajectory; MRC is the destination.

A naive MRM implementation like "stop immediately" is inadequate and potentially dangerous. A vehicle halting mid-intersection creates a collision hazard for cross-traffic. A vehicle stopping in a live highway lane invites a rear-end impact. The MRM must therefore include a brief window of *expanded* autonomous authority: the vehicle selects a safe pull-over location, executes lane changes or clears the intersection, activates hazard indicators, and comes to a controlled stop. This sequence may require 10 to 30 seconds of decision-making that, under normal operations, would involve the remote operator.

The MRM trigger is deterministic. When the measured communications blackout duration exceeds a predefined threshold *T* (calibrated to the operational design domain, typically on the order of seconds), the vehicle initiates the MRM autonomously. No real-time human approval is solicited or expected. The MRM itself functions as the system's safe default state, the fallback that the entire safety case is built around, consistent with the fault-tolerance principles of ISO 26262 [ISO, 2018].

## 3. The Autonomy Boundary: Pre-Delegation Framework

The critical insight is temporal. The human operator does not approve the MRM at the moment of crisis. They approved a *policy* during system design, validation, and deployment configuration. This is pre-delegation: the engineering and operational authority chain defines in advance the bounded set of actions the vehicle may take autonomously when communication is lost.

This yields a two-tier decision taxonomy:

- **Pre-delegated (system executes autonomously):** MRM initiation upon comms-loss detection, path-to-shoulder or safe-stop-location selection, hazard light and brake light activation, speed reduction profile, and final stop execution. These actions are bounded, validated, and terminate in a lower-risk state.
- **Requires human reconnection (system waits at MRC):** Resuming the original mission, rerouting around an obstruction discovered during MRM, handling novel edge cases outside MRM scope (e.g., a road closure at the planned pull-over point with no pre-validated alternative), and any action that would expand the vehicle's operational envelope beyond the MRM procedure.

The boundary criterion is explicit: a decision is pre-delegatable if and only if it is **(a)** bounded in scope and duration, **(b)** reversible or monotonically transitions the system toward a lower-risk state, and **(c)** was explicitly validated through simulation, formal verification, or operational testing at design time. Any action failing one of these conditions requires a human in the loop, which in turn requires communication, which means the vehicle must wait at MRC until reconnection.

This framework preserves human authority without depending on human availability. The operator's judgment is embedded in the policy, not demanded in the moment.

## 4. Trust and Latency: The Reconnection Handoff

When the data link is restored, the operator re-enters a system that has been acting without them. The handoff is safety-critical and must be designed for three distinct system states:

**State 1: MRC achieved.** The vehicle is parked and safe. Urgency is low. The operator reviews a post-incident summary and decides whether to resume the mission or dispatch roadside assistance. This is the clean case.

**State 2: MRM in progress.** The vehicle is actively maneuvering toward MRC. This is the most dangerous moment for intervention. The interface must unambiguously signal that the MRM is executing, display the planned trajectory and estimated time to MRC, and default to *not* transferring control. The operator should be required to perform an affirmative acknowledgment before overriding an in-progress MRM. Premature operator takeover during a validated maneuver sequence introduces the exact class of "automation surprise" errors documented extensively in aviation human-factors literature [Parasuraman & Riley, 1997].

**State 3: MRM not yet initiated.** The comms loss was below threshold *T* but the operator was still briefly unreachable. The vehicle is in normal operation but the operator has a gap in situational awareness. This is the failure-of-failure-mode: everything nominally worked, yet the operator must rapidly reconstruct context. The reconnection interface must provide a compressed state summary covering what events occurred during the blackout, the vehicle's current state, and the next required operator action. A raw telemetry log that demands minutes of parsing under time pressure is not acceptable.

Across all three states, one design principle holds: the system must communicate its own uncertainty honestly. An MRM path confidence of 87% with an identified alternative is more useful to the reconnecting operator than a binary "MRM OK" status. Calibrated confidence reporting allows the operator to make an informed decision about whether to intervene, rather than forcing a trust-or-don't binary.

## 5. Conclusion

Operator availability is a resource, not a guarantee. For L4 autonomous vehicles relying on remote teleoperation, safety-critical decisions will inevitably arise when that resource is unavailable. The correct HITL paradigm is pre-delegation: bounded, reversible, design-time-validated policies that the system executes autonomously when communication fails, terminating in a minimum risk condition where the system waits for human reconnection.

This pattern is not specific to autonomous vehicles. Any safety-critical automated system operating in a degraded-communications environment, whether undersea robots, remote surgical systems, or military autonomous platforms in contested electromagnetic environments, faces the same structural tension. Shift the locus of human authority from real-time approval to design-time policy specification, and build the system's autonomy boundary around what can be validated in advance.

---

## References

- **[ISO, 2018]** ISO 26262:2018, *Road vehicles — Functional safety*. International Organization for Standardization.
- **[Parasuraman & Riley, 1997]** Parasuraman, R., & Riley, V. (1997). Humans and automation: Use, misuse, disuse, abuse. *Human Factors*, 39(2), 230–253.
- **[SAE, 2020]** SAE J3018:2020, *Safety-relevant Guidance for On-Road Testing of Prototype Automated Driving System (ADS)-Operated Vehicles with a Remote ADS Operator*. SAE International.