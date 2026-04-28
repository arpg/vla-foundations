## ROBO 7000 HITL Safety Writeup

# Problem description:
 Robot navigation task in an extreme environment, high latency in terms of compute (i.e., relatively low compute, enough for a VLM but not much more), as well as signal connection with a ground station. For scenarios to assess, the robot will either be in a safe state or an unsafe state when losing connection with some form of grounding (base state, human rectification, etc.)

# Writeup:

Robotic autonomy in extreme environments requires systems to have characterizable failure modes. 
In scenarios where robustness is paramount and failure is not tolerated, system complexity is a luxury that has to be designed around and ultimately deterministic. 
Traditional methods, while unable to adapt to generalized uncertainty, are still chosen over modern methods because their uncertainty may grow over time; these processes are stochastically predictable. 
For example, a geometric pipeline might not have the semantic knowledge to assess a scenario, but we can put numbers behind when the system will fail, and rigorously determine how the system will fail when it does. 
Recent foundation models provide strong semantic generalization to out-of-distribution scenarios that we can not anticipate, but their difficult-to-characterize failure modes limit their deployment in mission-critical robotic systems. 
Most of the current research with foundation models has been centered in environments where we can simply try again - if it fails, we can re-form, retrain, and retest. 
As such, a lot of work has been centered on replacing the age-old systems with new, fancy models with increasingly more capabilities at the cost of increasingly more data. 
I think the solution to this problem is not to find a better way to replace the old methods, but to find a better way to integrate the new methods with the old methods.

My proposed framework is to keep a standard baseline robot navigation stack that relies on planning around typical sensor data, and to augment the system with an additional source of semantic guidance through a VLM-like agent. 
The vision-language alignments or outputs from the model should, in some way, inform the decision-making process of the lower-level autonomy stack, but not override or decide concrete actions for the robot itself. 
The issue with foundation models is their black box nature - we can not fully understand how and when their outputs happen, except that they happen with a relatively high accuracy, but we may not always understand the task at hand. 
As such, if we leave the models to only do what they are good at - generalized understanding, we can leave the more intricate processes to the traditional geometric baseline planners. 
For example, the models can provide semantic understanding of hazards through an RGB image that may not be easily picked up by the other sensors. 
This way, if a base station is not able to communicate at a high latency - or at all - with the robot, the growing local uncertainty regions of the baseline method can potentially be ‘rectified’ by the model. 

Additionally, in these blackout durations, the system should save a log of its reasoning passed into the pipeline, as well as a log of previous safe states (which can be verified in every instance we reconnect with the base station). 
This way, when we reconnect, humans can verify these safe states and reasonings, and when we disconnect, the robot can decide how to act if it is currently in an unsafe state or a safe state. 
In an unsafe state, we just try to roll back to previous safe states, and if in an unsafe state, we only act if the model has high certainty of semantic knowledge of the scene for the baseline method to act (i.e. default behavior). 

The ideal scenario would be for a foundation model to be able to know when it should ask a human for help, but given the constraints of the problem, I think the best we can do is have the baseline methods and the foundation models to keep each other in check. 
In the future, as we start to delegate more tasks to the model, some form of closed-loop reasoning (e.g., thinking models) that is able to fact-check itself would be interesting to implement, or some form of integration with more mathematically rigorous formalisms like linear temporal logic reasoning. 
Human-centric environments, like households, offices, or even factories, allow human operators to be able to check in on progress and monitor uncertainties - whether the robot is sure or not of its own uncertainties. 
In areas where humans can not always be readily available to help these robot agents, they need to be able to act on their own. 
My take is essentially that why have a robotic system that relies on itself, rather than two robotic systems that rely on each other.
