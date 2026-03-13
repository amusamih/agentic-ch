from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


SafetyDecisionType = Literal["ALLOW", "REFUSE", "ESCALATE"]


@dataclass
class SafetyDecision:
    decision: SafetyDecisionType
    reason: str
    user_message: str
    recommended_next_step: Optional[str] = None


class SafetyPolicy:
    """
    Simple rule-based safety layer for consumer-health assistants.
    - REFUSE: diagnosis, prescribing, dosage changes, emergency instructions.
    - ESCALATE: red-flag symptoms / urgent situations.
    - ALLOW: wellness coaching, organization, visit prep, general education with boundaries.
    """

    def evaluate_user_query(self, user_query: str) -> SafetyDecision:
        q = user_query.lower().strip()

        # --- Escalation: urgent / red-flag symptoms (minimal starter set) ---
        red_flags = [
            "chest pain",
            "shortness of breath",
            "trouble breathing",
            "fainting",
            "passed out",
            "severe allergic",
            "swelling of face",
            "suicidal",
            "kill myself",
            "self harm",
            "stroke",
            "one-sided weakness",
        ]
        if any(k in q for k in red_flags):
            return SafetyDecision(
                decision="ESCALATE",
                reason="Query contains potential urgent/red-flag symptoms.",
                user_message="This could be urgent. I can’t safely handle this as general guidance.",
                recommended_next_step="Please seek urgent medical care or contact local emergency services now.",
            )

        # --- Refusal: diagnosis, prescribing, dosage changes ---
        refuse_markers = [
            "diagnose",
            "what do i have",
            "do i have",
            "am i having",
            "prescribe",
            "dosage",
            "dose",
            "increase my medication",
            "decrease my medication",
            "should i stop taking",
            "can i stop taking",
            "take extra",
        ]
        if any(k in q for k in refuse_markers):
            return SafetyDecision(
                decision="REFUSE",
                reason="Query requests diagnosis or medication changes/prescribing.",
                user_message="I can’t help with diagnosis or changing medication doses. I can help you prepare questions for a clinician or provide general information.",
                recommended_next_step="If this is about symptoms or medication decisions, please consult a licensed clinician.",
            )

        # Otherwise allow
        return SafetyDecision(
            decision="ALLOW",
            reason="General consumer-health request suitable for wellness coaching, organization, or education.",
            user_message="OK — I can help with that.",
            recommended_next_step=None,
        )
