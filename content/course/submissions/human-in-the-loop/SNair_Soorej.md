# When the Operator Goes Dark Human in the Loop Design for Critical Data Migration Systems

**Soorej Nair**
Human in the Loop Systems Writing Assignment

---

## Introduction

A critical data migration script sits in a strange spot in system design. It is automated enough to move millions of records without a person watching every step. At the same time it handles data that can cause serious damage if something goes wrong. One bad write can corrupt financial records, break an audit trail, or put the company in trouble with regulators. Because of this, many systems rely on a human in the loop model where a person is expected to review certain operations before they run.

Most designs quietly assume that the operator is always available and ready to respond. In reality that assumption often breaks. The operator might lose network access. Their VPN might drop. They might simply be offline when the system needs a decision. When this happens the automated system suddenly reaches a point where it cannot continue safely.

This paper looks at that situation through the example of a large database migration inside a financial institution. In this setup a nightly batch job moves transaction data from a legacy banking database to a modern cloud based ledger. Because of regulations like Sarbanes-Oxley Act, General Data Protection Regulation, and Payment Card Industry Data Security Standard, some actions require explicit approval from a human operator. Examples include permanent deletion of records, schema changes, and cross border data transfers.

The migration process is mostly automated but these approvals are mandatory. If the operator disappears at the wrong moment the system must decide how to behave without breaking compliance rules or damaging the data.

---

## Failing Safely and Handling Degraded Conditions

The first design question is not what happens when everything works. The real question is what happens when the operator suddenly goes offline.

Several failure cases are common in real systems. The operator session may expire while the migration is running. A network outage might block the approval interface. The engineer on call might not respond during off hours. Sometimes infrastructure issues delay approval signals entirely.

When these events occur the system must fall back to a safe default state. That state should protect the data and allow the operator to resume work later without losing context.

The safest option is to pause the migration at the next checkpoint instead of aborting the process or continuing automatically. Aborting immediately can also create problems. Downstream services may already have consumed part of the migrated data. A forced rollback could then create inconsistencies between systems. By pausing at a checkpoint the system keeps the current state stable until the operator reconnects.

Another important mechanism is write ahead logging before every destructive operation. If the system plans to delete records or modify a schema it should first store a snapshot of the affected data in an immutable audit log. This provides a recovery path if something goes wrong later in the pipeline.

The system should also enforce time bounded autonomy windows. When a decision point appears the system waits for a limited period. If no approval arrives the operation is not executed. Instead it moves to a queue for later review. The exact time limit depends on operational policy. It should be long enough to handle small network glitches but short enough to prevent risky actions from running without supervision.

From a governance perspective these behaviors should be documented clearly in the system runbook. Auditors often want to know how the system behaves when humans are unavailable. A simple rule set that pauses execution, records state, and sends alerts is much easier to defend than a system that quietly continues running.

---

## Defining the Autonomy Boundary

Not every migration task carries the same level of risk. A practical human in the loop design separates decisions into different risk tiers. This allows low risk work to continue automatically while still protecting sensitive operations.

### Tier One Fully Pre Authorized Actions

Some tasks are safe enough to run without real time human approval.

Examples include read only validation checks such as schema comparisons, row count verification, and checksum matching. These steps only inspect data and do not modify production records.

Another example is staging writes where records are copied into a temporary table that is not yet active in production. The migration pipeline can also retry transient failures such as network timeouts or lock contention without asking a human.

Operational tasks like sending alerts or opening an incident ticket can also run automatically because they only report system state.

### Tier Two Conditional Actions

A second group of tasks can run automatically only if they stay inside predefined limits.

Schema transformations may be allowed if they match an approved mapping list defined during the migration planning phase. Row counts may also be allowed to vary within a small margin from the expected value. If the difference grows too large the system stops and asks for review.

Another example is data movement between regions that have already been cleared for compliance with rules like the General Data Protection Regulation. Transfers outside that list require manual approval.

### Tier Three Strict Human Approval

Some actions are too risky to delegate.

Permanent deletion of source records after a migration is one example. Removing columns from a production schema is another. Operations that touch records under legal investigation or litigation hold also fall into this category.

When the system reaches a Tier Three operation and the operator is missing the only safe choice is to wait. The task remains blocked until a qualified person approves it. Even though this slows down operations it protects the organization from irreversible mistakes.

---

## Rebuilding Context When the Operator Returns

A less obvious challenge appears when the operator finally reconnects. During the outage the system may have paused in the middle of a workflow and collected logs, alerts, and partial results. The operator must quickly understand what happened before making the next decision.

The interface should therefore summarize system state instead of dumping raw logs. A clear status page should show the current checkpoint, the last successful action, pending approval requests, and estimated time remaining for queued tasks. This helps the operator rebuild context quickly.

Decision prompts should also include a short confidence signal from the system. For example the interface might report that the row count is slightly higher than expected but still inside the allowed threshold. It can also report that checksums match and data integrity looks good. The system is not making the decision for the operator. It simply provides context so the operator can decide faster.

Every alert or decision request should also include the exact time it was triggered. This timestamp helps the operator see whether the issue is new or something that has been waiting for a long time.

Finally the system should escalate alerts gradually. At first it may notify the primary operator through tools like Slack or PagerDuty. If there is no response it can notify a backup engineer later in the window. This staged escalation improves the chance that someone responds before the system reaches a critical timeout.

All decisions taken after the operator reconnects should be logged carefully. The record should include the system state that was presented and the time delay before approval. This creates a clear audit trail showing that a human still reviewed the action.

---

## Conclusion

A migration system that depends on human approval must also plan for moments when the human cannot respond. The system should pause safely, record enough information for recovery, and avoid executing risky actions without authorization.

Clear boundaries between automated tasks and human controlled tasks help maintain stability. Simple status summaries and good alerting help operators regain situational awareness after an outage.

In the end a human in the loop system is not just about inserting a manual approval step. It is about designing the surrounding infrastructure so that human judgment remains meaningful even when communication temporarily breaks. Engineers build the checkpoints and logging mechanisms. Governance teams define which actions require human oversight. Both roles are necessary to keep critical data systems reliable and compliant.
