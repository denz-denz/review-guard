import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Thresholds:
    advertisement_auto_remove: float = 0.90
    irrelevant_graylist: float = 0.75
    rant_hold_review: float = 0.70
    relevancy_min_pass: float = 0.50

    @staticmethod
    def from_file(path: str) -> "Thresholds":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return Thresholds(**{**Thresholds().__dict__, **cfg})

def decide(th: Thresholds, probs: Dict[str, float], relevancy: float):
    """Return decision dict with action and reasons."""
    reasons = []
    action = "pass"

    if probs.get("advertisement", 0) >= th.advertisement_auto_remove:
        action = "remove"
        reasons.append("High-confidence advertisement")

    if action == "pass" and probs.get("irrelevant", 0) >= th.irrelevant_graylist:
        action = "graylist"
        reasons.append("Likely irrelevant content")

    if action == "pass" and probs.get("rant_without_visit", 0) >= th.rant_hold_review:
        action = "hold"
        reasons.append("Possible rant without visit")

    if action == "pass" and relevancy < th.relevancy_min_pass:
        action = "graylist"
        reasons.append("Low relevancy score")

    return {"action": action, "reasons": reasons}
