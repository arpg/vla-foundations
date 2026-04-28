# Human-in-the-loop integration of AIS data into maritime autonomy

Most maritime vessels are equipped with an automated idenfication system (AIS), used to broadcast status and position announcements to nearby vessels and other receivers. AIS messages are sent often (with a frequency between seconds and minutes) and consist of one of 27 predefined message types, from position information to safety messages to manually entered voyage data. AIS is used by vessel officers and maritime authorities to avoid collisions, aid navigation, and monitor fleets, particularly far from the coast.

Automation and decision support in such vessels potentially involve many humans in many autonomy loops. I found the very specific caser of AIS interpretation interesting because the obstacles to complete autonomy are unusual: 

* AIS message fidelity relies solely on the cooperation of vessels, governed by complex maritime law. There is little guarantee that any particular AIS message is accurate, or that it will exist at all.
* AIS data is both heterogenous and atomic: messages can consist of anything from numeric geospatial data to extended, natural language route descriptions, both necessitating further reasoning to extract useful operational conclusions.
* AIS supplements other sensor data, but can contain qualitative, non-numerical information that cannot be gleaned from other sources, depending on what the source vessel chooses to broadcast. 
* AIS is not secure and [vulnerable to many forms of attack](https://ieeexplore.ieee.org/abstract/document/10411879/).
* AIS data is difficult to label exhaustively. While advancements in language modeling have helped, the problem of determining a vessel's intent from its motion [is nontrivial](https://dl.acm.org/doi/pdf/10.1145/3748636.3762805). 

As such, while expert-quality insights from AIS [possible to compute](https://www.sciencedirect.com/science/article/pii/S2468502X21000401), doing so live, during a vessel's transit, poses a number of edge case risks best ameliorated **with a human in the loop**.

#### Disclaimer

I am not a maritime autonomy expert - this is a "writing and thinking" exercise, not a breakdown of the state of AIS usage in maritime autonomy. From what I can tell, the dominant approach is to consider AIS data secondary, if it is used at all. This seems to be the tactic taken by maritime autonomous surface ships currently in development.

That said, there does seem to be a place for computerized AIS interpretation in semiautomated vessels. 


## Default state
**Key question**: *"How should the automated system be designed to fail safely?"*

Most of the time, small interruptions in AIS interpretation are unproblematic. AIS messages rarely contain information that cannot be collected from other sensor data, and the safe behavior is to proceed as normal. To put it a different way, the default state of a vessel should rely primarily on primary-source [sensor] data, rather than secondary-source AIS data.

However, some AIS messages are important. For instance, consider an AIS message 12 "Safety Related Message," which may take many difficult-to-interpret forms (natural language, commercial codes, encrypted) and could contain information such as reports of a malfunction on a nearby vessel. A message such as this ought to be flagged for human intervention. If no human intervention is available, a variety of default vessel behavior could be considered, but as far as AIS itself is concerned, the vessel's inability to interpret the information must be communicated.


## Communication, remoteness and latency
**Key question**: *"How [do] system latency, extreme environments, or communication failures change the requirements for a 'human in the loop?'"*

The International Maritime Organization [defines](https://web.archive.org/web/20190530173351/http://www.imo.org/en/MediaCentre/MeetingSummaries/MSC/Pages/MSC-100th-session.aspx) four degrees of maritime autonomy. Of primary interest to us are degree one "automated processes and decision support" (with humans still present) and degree three "remotely controlled ship" (with no humans aboard), as they pose significantly different challenges under circumstances where a human is required to interpret an AIS message.

Under "degree one," human intervention is expected - the autonomy is a decision support system. Under this circumstance it is reasonable to wait for the input of a human operator (and, though I cannot be certain, I suspect most semiautomated vessels always require human intervention to bridge the gap between AIS data and system action). Under these circumstances, we require a _constant_ human presence to guarantee AIS interpretability.

However, unlike an autonomous passenger vehicle (keyword _passenger_), cargo vessels need not _de facto_ have any humans on board at all. Under "degree three," a human operator may not be present, and since cargo ships may operate in extremely remote locales, none may be readily available. Even with satellite connectivity, bandwith and reliability may not be suitable for the amount of data produced by a maritime vessel to reach an onshore human expert. Fortunately, the locations where careful AIS parsing are most often necessary - congested ports, straits, and canals - are also those where data links are most often reliable. 

## Trust and cognitive load
**Key questions**: *"How do you manage the human's cognitive load when they suddenly regain connection and have to make a rapid decision based on a backlog of system states?"*

There are several approaches on this front.

* **Ignore**: Positioning and navigational information, which can be conveniently rendered, are easier to visually parse to a human than natural language in AIS messages. Even if failing to interpret AIS messages leads to an illegal (e.g. near-collision) state, if the state can be detected with other sensor information, the AIS information should be discarded or postponed.
* **Important first**: Some AIS message types are more important than others, and should be presented first.
* **Recent first**: Uninterpretable messages which were considered important upon their receipt may not be important at the time a data link is reestablished.

AIS interpretation is very distant from the control level, in contrast to many human-in-the-loop systems where a human takes complete and direct control of the system. Specifically, human intervention takes place specifically in the sensing architecture rather than over the entire system. As such, in the scenario described here, presenting information to the human in the loop efficiently is not sufficient if the resulting decision making cannot enter the system fast enough to be useful. 

## The autonomy boundary
**Key question**: *"Exactly which decisions can be pre-delegated to the system to handle on its own, and which must wait for a human operator to reconnect, regardless of the delay?"*

The "difficult cases" for AIS interpretation are highly contextual, given the indefinite amount of information an AIS message can contain. Some circumstances imply that an AIS interpretation failure may not be severe enough to merit stalling behavior:

* Partially failed interpretations from ships which have been localized with other equipment
* Failed interpretations on AIS message 5 "Static and voyage related information," which do not typically contain time-sensitive information

In other cases, a human operator must be reached before the message can be safely discarded:
* Messages in congested areas (which, aside from being more difficult to navigate, also make route information more useful)
* Search-and-rescue and safety-related messages (potentially indicating dangerous circumstances or "crucial, non-emergency" messages)
* Cases where the message information clearly does not match other sensor information (potentially inficating an [intentionally misused broadcast](https://en.wikipedia.org/wiki/Automatic_identification_system#Spoofing)).