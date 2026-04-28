# Human in the Loop System Design When Operating in Extreme Environments

*Zack Allen*

## Introduction

The design of human-in-the-loop systems becomes significantly more complex when the operating environment introduces latency, communication blackouts, and physical inaccessibility. Few scenarios illustrate this as clearly as NASA's next-generation Lunar Terrain Vehicle (LTV). The LTV is expected to survive and operate autonomously on the lunar surface for approximately three years before astronauts arrive. Once crew members are present, it must support driver-assisted operation while allowing full manual override. This progression from autonomous to human-directed control — across an environment defined by communication delays, radiation, and unpredictable terrain — makes the LTV an ideal case study for examining how extreme conditions reshape human oversight requirements.

## 1. Failing Safely

The failure modes confronting the LTV fall into two categories: *environmental state failures* and *system failures*. Environmental failures include loss of power and degraded communications. The Earth–Moon link carries a round-trip delay of roughly three seconds under ideal conditions, and this can grow unpredictably. System failures encompass sensor degradation from ionizing radiation, perception errors caused by the feature-scarce lunar landscape, localization drift, and control instability from one-sixth gravity and loose regolith.

The default safe state must therefore be context-dependent. During uncrewed operation, the overriding priority is vehicle preservation. If communication is lost, the system must maximize its ability to remain charged and functional, even at the cost of task completion. When astronauts are present, the hierarchy inverts: crew safety and mobility take absolute precedence. These two regimes require a mode-dependent chain of command managed internally by the vehicle.

To support both modes, the vehicle must maintain an up-to-date health vector describing all sensor and actuator states. As capabilities degrade from radiation, dust, and wear, the system must progressively constrain its operational envelope, restrict available commands, and always retain a safe abort option when navigating unpredictable terrain.

## 2. The Autonomy Boundary

Before autonomous operation can begin, the vehicle must establish a known-safe region. Upon delivery, it should be closely monitored by teleoperators as it maps the area around its landing site. This supervised phase produces a high-confidence local map — a "home base" for all subsequent autonomous expansion.

**Fully automated capabilities.** Two functions must be automated without exception. First, power management: the vehicle must dynamically power components on and off to optimize survival, a "fight or flight" response necessitated by the absence of any repair crew. Second, communication link maintenance: the system must autonomously adjust behavior to prevent loss of signal, and if a global blackout occurs, sustain itself until contact is reestablished.

**Conditionally autonomous operations.** Navigation beyond the safe zone is the gray area. Exploration requires confident localization at all times. Losing the vehicle's position while astronauts are on the surface could be fatal. The system must define localization and terrain confidence thresholds, halting and requesting operator input when either drops below acceptable levels. Terrain detected outside the vehicle's operational envelope must require explicit human override before traversal.

**The vehicle's right to override.** An important inversion arises during uncrewed operation: the vehicle should have authority to override human commands if executing them would risk rendering it inoperable. This manifests as an autonomous return-to-home capability, where the machine overrides the human in the interest of long-term mission success.

## 3. Trust and Latency

When the nominal three-second delay stretches to ten, twenty, or a hundred seconds, the system must prevent operators from making rash decisions upon reconnection. This requires a pre-agreed definition of loss of signal (LoS) so both vehicle and ground station can independently transition into degraded-communication protocols.

**Prioritized telemetry on reconnection.** The vehicle's first transmission should be a concise health summary. The ground station should pre-compute expected values from system models, presenting operators with a predicted-vs-actual comparison rather than raw data, with ergonomic color coding on all fields. Next, the vehicle should report estimated displacement since LoS, accompanied by imagery of its surroundings. These two elements — health and situational awareness — are sufficient for rapid assessment.

**Supporting context.** The system should also maintain a task log organized in a pre-agreed format (explore, recharge, find communications, hibernate), rank anomalies by severity with only critical ones in the initial packet, and present proposed next actions to accelerate the operator's return to situational understanding.

**Emergency intervention.** When telemetry is restored and the vehicle is in an actively dangerous state — barreling toward a cliff, for instance — the operator cannot be expected to diagnose and respond in time. The solution is an "abort all movement" command: a single button, always available, guaranteed to bring the vehicle to a safe stop. This abort capability, integrated into the autonomy stack, forms the third essential component of the operator interface alongside the health summary and situational display.

## 4. Conclusion

The LTV demonstrates how extreme environments reshape human-in-the-loop design. The conventional assumption that a human operator is always available and authoritative breaks down when communication is delayed or absent. In its place, a more nuanced framework emerges: autonomy expands and contracts based on conditions, the system can override human commands to preserve operability, and the interface protects operators from cognitive overload. The three principles explored here: context-dependent safe defaults, confidence-bounded autonomous expansion, and prioritized telemetry with an ever-present abort capability provide a generalized framework for a space system where the human in the loop cannot be assumed to be continuously connected.