from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from openenv.core import Observation as OpenEnvObservation
from openenv.core import State as OpenEnvState


class Action(BaseModel):
    action_type: Literal["route", "request_info", "escalate", "archive", "defer"] = Field(
        ...,
        description=(
            "Action to take on the current email: "
            "'route' -> send to the correct department; "
            "'escalate' -> immediate escalation for angry/threatening customers; "
            "'request_info' -> ask the sender for more details when the email is too vague; "
            "'archive' -> discard spam, test messages, or irrelevant content; "
            "'defer' -> pass to a human reviewer when uncertain (small cost, avoids large mistakes)."
        ),
    )
    department: Optional[Literal["billing", "sales", "technical_support", "legal_compliance"]] = Field(
        None,
        description=(
            "The department to route to. Required only when action_type is 'route'. "
            "legal_compliance handles GDPR erasure, SOC2 audits, and security incidents."
        ),
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Agent's confidence in this action (0.0-1.0). Optional — defaults to 0.5. "
            "Overconfident wrong answers (>0.8 confidence, wrong action) receive extra penalty. "
            "Tracked as calibration_score (Brier) in state."
        ),
    )


class ThreadMessage(BaseModel):
    role: Literal["customer", "support"] = Field(..., description="Who sent this message.")
    body: str = Field(..., description="Message content.")


class Email(BaseModel):
    id: str = Field(..., description="Unique email identifier.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Email body text.")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        "medium",
        description="Urgency tier. Affects reward: critical=2.0x, high=1.5x, medium=1.0x, low=0.5x.",
    )
    sender_tier: Literal["free", "pro", "enterprise"] = Field(
        "free",
        description="Customer subscription tier. Enterprise triggers stricter SLA rules.",
    )
    customer_context: Optional[str] = Field(
        None,
        description="Brief history of prior interactions, if available.",
    )
    queue_position: int = Field(0, description="Position in the queue (0-indexed).")
    is_consequence: bool = Field(
        False,
        description="True if dynamically injected as a consequence of prior mishandling.",
    )
    thread_history: Optional[List[ThreadMessage]] = Field(
        None,
        description=(
            "Prior messages in the same conversation thread, if available. "
            "Correct triage may require reading the full thread, not just the current message."
        ),
    )
    sla_deadline_minutes: Optional[int] = Field(
        None,
        description="Explicit SLA deadline in minutes for critical/high emails. Visible to agent.",
    )
    consequence_stage: int = Field(
        1,
        description="Consequence escalation stage (1=angry, 2=social media threat, 3=legal/chargeback).",
    )


class Observation(OpenEnvObservation):
    """Full observation returned by reset() and step()."""
    current_email: Optional[Email] = Field(None, description="The email to triage. Null when backlog empty.")
    backlog_size: int = Field(..., description="Emails remaining in queue (including current).")
    previous_action_feedback: str = Field(..., description="Human-readable feedback from previous action.")
    reward: float = Field(0.0, description="Priority-weighted step reward (can be negative).")
    done: bool = Field(False, description="True when episode is finished.")


class State(OpenEnvState):
    """Rich trajectory-level metrics."""
    emails_processed: int = Field(..., description="Emails actioned so far.")
    total_emails: int = Field(..., description="Total emails (may grow from consequences).")
    current_score: float = Field(..., description="Cumulative clamped score (0.0, 1.0).")
    missed_escalations: int = Field(0, description="Escalation opportunities missed.")
    escalation_recall: float = Field(0.0, description="Fraction of escalation emails correctly escalated.")
    false_escalation_rate: float = Field(0.0, description="Fraction of non-escalation emails incorrectly escalated.")
    sla_compliance: float = Field(0.0, description="Fraction of critical/high emails handled correctly.")
    routing_accuracy: float = Field(0.0, description="Fraction of route-target emails correctly routed.")
    archive_accuracy: float = Field(0.0, description="Fraction of spam/irrelevant emails correctly archived.")
    deferred_count: int = Field(0, description="Emails deferred to human review.")
    consequences_triggered: int = Field(0, description="Consequence follow-ups injected by mishandling.")
    escalation_budget_remaining: int = Field(0, description="Escalation tokens remaining.")
    customer_satisfaction: float = Field(1.0, description="CSAT meter (1.0->0.0). Decays on mishandling.")
    churn_risk: bool = Field(False, description="True when CSAT dropped below 0.3, triggering early termination.")
    calibration_score: float = Field(0.0, description="Brier score measuring confidence calibration (lower=better).")
    consequence_stage_reached: int = Field(1, description="Highest consequence stage triggered (1-3).")
