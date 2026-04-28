# Human-in-the-Loop Teleoperation for Autonomous Vehicles Under Communication Failure

**Lorin Achey**

## Introduction

The safety case for many autonomous vehicle (AV) deployments rests on a comforting assumption that when the vehicle encounters a situation beyond its operational design domain (ODD), a remote human teleoperator can seamlessly assume control. Companies like Zoox, Waymo, and Cruise have built entire operational layers around this premise by staffing remote assistance centers with trained operators who monitor fleets and intervene when an AV signals distress. But this architecture embeds a fragile dependency. Cellular networks drop. Redundant links fail simultaneously during infrastructure outages. Urban canyons and tunnels create dead zones. The very edge cases that demand human judgment are often correlated with the conditions that sever the communication link. There is also the potential for adversarial attacks on communicaion links which introduces additional vulnerability.

This write-up examines what happens when the teleoperator connection breaks during a critical moment, and some considerations for how the system can be designed to handle that failure.

## Failing Safely: Designing for Disconnection

The fundamental question when a remote operator is cut off is: what should the vehicle do by default? The answer cannot be "continue driving normally," because the vehicle has already flagged that it needs help. But it also cannot always be "stop immediately," because an abrupt halt on a highway is itself a safety hazard.

This means that whatever default strategy exists must be context-aware and represent minimal risk. These can be referred to as Minimal Risk Conditions from here on (MRCs). The vehicle could carry a pre-computed library of safe fallback maneuvers indexed by its current operational context:

- **Highway / high-speed**: Gradually decelerate, activate hazard lights, and pull to the nearest shoulder or breakdown lane. The vehicle would have to use its remaining perception stack to execute this maneuver autonomously, even if the perception confidence is degraded.
- **Urban / low-speed intersection**: Come to a controlled stop in the current lane, yielding to immediate traffic, while broadcasting intent (hazards, exterior displays) to surrounding road users.
- **Parking lot / depot**: Simply stop in place.

A critical design principle/assumption introduced here is that the default state must be a achieved through action, not passivity. A vehicle that simply freezes its last control inputs is dangerous. The fail-safe must be an active, deliberate maneuver toward a stable state (even if that means remaining in place but switching the hazards on). This means the planner must be an independent, high-reliability subsystem, potentially running on a separate compute partition with its own validated perception pipeline, so that it remains functional even when the primary autonomy stack is compromised.

Additionally, the system should implement a heartbeat protocol with the remote operations center. If the vehicle does not receive an operator heartbeat within a defined timeout (e.g., 3–5 seconds for urban driving, 1–2 seconds for highway speeds), it should autonomously initiate its maneuver without waiting for an explicit command. The timeout must be calibrated to the kinematic reality: at 65 mph, a vehicle covers roughly 30 meters per second, so even a few seconds of indecision translates to substantial uncontrolled travel. The risk with this idea is that it could take some tuning to produce appropriate behavior.

## The Autonomy Boundary: Responsible Delegation

Not all decisions carry equal risk, and a well-designed HITL system should pre-delegate authority along a clearly tiered spectrum. Outlined below is a proposed three-tier framework:

### Tier 1: Fully Pre-Delegated (No Operator Needed)

These are decisions the vehicle must always be empowered to make on its own, because waiting for human input would itself create danger:

- Collision avoidance and emergency braking
- Execution of the MRC maneuver upon communication loss
- Yielding to emergency vehicles
- Basic lane-keeping and car-following within the ODD

These actions are safety-critical and time-critical. Requiring a human in the loop here would violate the fundamental latency constraints of vehicle dynamics.

### Tier 2: Conditionally Delegated (Operator Preferred, Autonomy Permitted After Timeout)

These are decisions where human judgment adds value, but where indefinite waiting is unacceptable:

- Navigating past a double-parked delivery truck by briefly crossing a lane line
- Proceeding through a construction zone with ambiguous signage
- Handling a novel but non-emergency interaction (e.g., a traffic officer waving the vehicle through)

For Tier 2 situations, the system should request operator input and wait for a bounded period (perhaps 15–30 seconds at low speed). If the operator is unreachable or does not respond, the vehicle may execute a conservative autonomous decision, or if confidence is too low, escalate to an MRC maneuver (the pre-defined library of maneuvers introduced previously). The key would be that the timeout and fallback action are defined in advance through rigorous scenario analysis, not improvised at runtime. While this introduces risks from needing to tune these fallback actions, it might be easier to evaluate their behavior in simulation rather than the dynamically changing timeouts.

### Tier 3: Operator-Required (Wait Indefinitely)

These are decisions where the consequences of an incorrect autonomous action are severe enough that the vehicle should not proceed without human authorization, regardless of delay:

- Resuming operation after a safety-critical fault (e.g., sensor failure followed by repair)
- Entering a geofenced restricted zone
- Overriding a regulatory constraint (e.g., proceeding past a road closure)

In Tier 3 situations, the vehicle achieves its MRC and simply waits until it can reestabilsh the remote connection and receive operator instructions or a physical safety team shows up on site. The design philosophy here would be that some delays are preferable to some mistakes.

This tiering must be documented, reviewed, and version-controlled as rigorously as the software itself. It represents the operational policy of the system, and regulators, insurers, and the public all have a stake in where the boundaries are drawn. It also presents an auditable artifact for any transportation authority (which may be a pro/con depending on the situation).

## Trust and Latency: Operator Challenges

When the communication link is degraded rather than fully severed the HITL problem becomes subtler because it introduces latency in operator commands that could cause unpredictable vehicle behavior and thus confusion for the operator.

### Communicating Uncertainty Under Delay

A teleoperator viewing a 4-second-old video feed is not seeing reality; they are seeing the recent past. The system must make this temporal gap obvious, not buried in a status bar. Effective approaches might include:

- A prominent, color-coded latency indicator (green/yellow/red) that scales with the danger of acting on stale data
- Ghost overlays showing predicted current vehicle and obstacle positions extrapolated from the delayed feed
- Automatic restriction of operator authority as latency increases. For example, at >2 seconds of delay, the operator can issue high-level route commands ("pull over at next safe opportunity") but cannot directly steer the vehicle. This is perhaps the most controversial.

The goal is to prevent the operator from developing false confidence in their situational awareness. Latent data is dangerous precisely because it looks like current data.

### Managing Reconnection Overload

Perhaps an underexplored failure mode is what happens when the link is restored after a period of disconnection. The operator could suddenly be presented with a backlog: the vehicle has been operating autonomously, made several Tier 2 decisions, accumulated a queue of deferred Tier 3 requests, and the world state has changed. The cognitive load is enormous, and the pressure to "catch up" creates a prime environment for human error. One possible method is triage, similar to emergency room doctors, where the highest priorities are pushed to the top. Those must be knocked out before the next tier of requests are handled.

The system could manage reconnection through structured re-briefing. This might include a triage system as mentioned above but it could present the operator with a reliable interface to summarize the state:

1. Prioritized summary: Present the operator with a ranked list of pending decisions and state changes, not a raw chronological log. The most time-sensitive items surface first.
2. Decision scaffolding: For each pending item, provide the system's recommended action and confidence level, so the operator is reviewing and approving rather than reasoning from scratch.
3. Temporal buffer: Do not immediately restore full operator authority upon reconnection. Allow a brief orientation period (10–20 seconds) where the operator can review the summary before the system begins routing decisions to them.
4. Graceful handoff: Transition control back incrementally if the situation does not prevent immediate safety concerns rather than dumping the full cognitive load at once.

This idea comes from aviation crew resource management, where pilots returning from a break receive a structured briefing from the pilot who maintained control, rather than simply grabbing the yoke. There may be a similar "hand-off" checklist for emergency medicine and surgery when transitioning control to different members of the surgical staff. Both of these industries would make for good case studies.

## Conclusion

The reliability of the human-in-the-loop is bounded by the reliability of the link that connects them. Designing a HITL system that only works when communication is perfect is not designing a safe system. It is designing a system that is safe only under favorable conditions which is a poor assumption in the real world. True robustness requires that we treat the operator as a valuable but intermittent resource: available most of the time, but never guaranteed.

This reframing has architectural consequence. The vehicle cannot be designed as a remote-controlled machine with some autonomy bolted on. It must be designed as an autonomous machine that benefits from human oversight when available. The autonomy boundary framework, the context-aware MRC library, and the structured reconnection protocol are all expressions of this same principle: the system must be complete and safe on its own, and the human makes it better.
