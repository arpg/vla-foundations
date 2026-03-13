# When the Loop Breaks: Human-in-the-Loop Design for Autonomous Field Robots in Degraded Environments

**Kali Hamilton**

## Introduction

Most HITL system designs assume the human operator is always reachable, that connectivity is reliable, latency is low, and intervention can happen in real time. In practice, this assumption breaks constantly. During my time as a Field Robotics Engineer at Scythe Robotics, I helped scale an autonomous mowing fleet from roughly 10 to over 100 robots deployed across 18 states. The Scythe M.52 is a commercial autonomous mower that operates outdoors in unstructured environments (parks, campuses, sod farms, corporate landscapes) where cellular connectivity, GPS/RTK positioning, and even physical access to the machine are unreliable by default. The robot could only operate autonomously when both cellular and RTK connections were active; without them, it would E-stop in place. That single constraint ended up shaping almost everything about how we thought about failure, autonomy boundaries, and what operators actually needed from the system when things went wrong.

## Failing Safely: The Default State Problem

At Scythe, the fail-safe default was straightforward: if the robot lost cellular or RTK connectivity beyond a roughly 10-second dropout threshold, autonomous operation stopped and the machine E-stopped. Within that 10-second window, the robot could fall back on visual odometry to maintain localization, but beyond it, the system would not continue operating without a human in the loop.

An autonomous mower with spinning blades in public spaces, near children, pets, and landscaping crews, cannot afford to guess. The cost of continuing to mow when you've lost localization (driving into a ditch, hitting an obstacle you can no longer see around) far outweighs the cost of just stopping and waiting.

But this conservatism had real operational costs. Robots deployed in areas with poor cellular coverage or near tall buildings would E-stop frequently, frustrating operators and killing "blades-on time," which was the metric that actually mattered for customer value. We mitigated some of this by adding backup RTK correction services to cover dead spots. But environmental factors like solar flares disrupting GPS or tree canopy blocking RTK corrections were just outside our control. You can't fix the ionosphere. A big part of the job was honestly managing customer expectations about where autonomous operation would and wouldn't work reliably.

One thing that helped: the robot could still be run *manually* without connectivity. Even when autonomy wasn't available, the machine was still a mower. This kept it useful and kept the human in the loop, just as a direct operator instead of a remote supervisor.

## The Autonomy Boundary: Expanding It Responsibly

Scythe built a four-tier support model as the fleet scaled: Tier 0 (self-service documentation), Tier 1 (customer support), Tier 2 (field robotics engineering and field service), and Tier 3 (design engineering). Each tier was added sequentially as the company grew, and they ended up defining a natural hierarchy for which decisions the system could make on its own versus which needed a human.

**What the robot handled on its own:** Path planning, mow execution, and obstacle avoidance for objects it could confidently identify. These are high-frequency, low-consequence decisions where the cost of a minor error like an unnecessary stop-and-resume is just lost mowing time.

**What required a human:** Any obstacle that fell below the identification confidence threshold. The trickiest case was dynamic obstacle classification. The system flagged anything it thought was a moving object (humans, vehicles) as a hard stop requiring HITL clearance if that object appeared inside the mow zone but wasn't actually moving. This made sense from a safety perspective because a stationary child should still be treated as a hazard. But it produced painful false positives.

The best example: during testing on a sod farm with giant irrigation pivots, the robot kept classifying the pivot wheels as trucks. The pivots weren't moving, but the model tagged them as dynamic obstacles, so the robot would stop and demand human intervention over and over. It made autonomous operation basically impossible for the test team. The fix wasn't to lower the safety threshold. It was to retrain the perception model to tell the difference between farm pivots and trucks.

That experience stuck with me. The autonomy boundary should expand by making the system smarter, not by making it less cautious. When the robot couldn't tell a pivot from a truck, the answer wasn't "let it ignore potential trucks." It was "teach it what a pivot looks like." We tracked this kind of progress through MTBF (Mean Time Between Failures) metrics. If reliability was improving on a specific failure mode, that was evidence the system had earned more autonomy in that area.

The run-in program worked the same way at the hardware level: every robot logged 10-20 hours of operation before shipping to a customer, which caught about 80% of early failures. We were essentially proving the machine could handle basic operation before putting it in the hands of a less-expert operator.

## Trust and Latency: The Reconnection Problem

When connectivity dropped and came back, the human operator faced the "backlog problem": the system had been accumulating state changes and events during the disconnection, and now someone had to make sense of it all at once.

At Scythe, our diagnostic data was primarily ROS logs and bag files that had to be offloaded from the robot for playback. This was slow, competed with the robot's operational connectivity, and usually had to wait until the machine wasn't being actively used. In practice, that meant diagnosing a field event could take hours or days. The gap between "something happened" and "we understand what happened" was one of the hardest problems we dealt with.

As the fleet scaled, data generation exploded to around 2 billion logged rows. Even when engineers could access the data, finding the signal in the noise was genuinely difficult. We started monitoring data generation itself to make sure what we were logging was actually relevant and necessary, but I wouldn't say we ever fully solved this one.

The most concrete lesson about operator trust came from the UI. Early error messages were written by engineers for engineers: "ESTOP: Lizard Heartbeat Lost" referenced Lizard (lizard.dev), an internal framework managing connections between the main CPU and seven different embedded control systems. For the landscaping crew operators actually using the machine, this meant nothing. It could be any of seven different hardware connection failures, and the operator had no way to know which or what to do about it.

We eventually changed it to: "Hardware failure. Please attempt power cycle. If issue persists, contact customer support." That gave the operator the one action they could take on their own (power cycle) and, if that didn't work, connected them to Tier 1 support, someone who could potentially diagnose the issue remotely and get them back to mowing faster.

The broader point: error messages and system status should be written for the person who's actually going to read them, not for the person who wrote the code. A landscaping operator who just regained connection to a stopped robot doesn't need a fault tree. They need to know: can I fix this, and if not, who do I call? The tiered support model was really a framework for routing problems to the right human at the right level of detail.

## Conclusion

The continuously available, fully informed human operator doesn't exist in the field. Connectivity drops, data arrives late, and the humans in the loop range from expert engineers to operators who just started using the machine last week. Working at Scythe taught me that you need a conservative default state. Stop rather than guess when the loop breaks. You need an autonomy boundary that expands because the system got better, not because you relaxed the rules. And you need to think about what the actual human on the other end can do with the information you're giving them. The "loop" in human-in-the-loop isn't binary. It's a spectrum of connection quality, operator expertise, and support structure, and the system has to work at every point along it.
