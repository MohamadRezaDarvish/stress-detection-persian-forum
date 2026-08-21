
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import exp, log
from pathlib import Path
from typing import Optional


def retained_weight(elapsed_minutes: float, half_life_minutes: float) -> float:
    if elapsed_minutes <= 0:
        return 1.0
    return exp(-log(2.0) * elapsed_minutes / half_life_minutes)


@dataclass
class UserRiskState:
    last_timestamp: Optional[datetime] = None
    last_alert_timestamp: Optional[datetime] = None
    fast_score: float = 0.0
    sustained_score: float = 0.0
    baseline_score: float = 0.0
    recent_classes: list[str] = field(default_factory=list)
    post_count: int = 0

    def update(self, timestamp: datetime, score: float, clinical_class: str, config: dict):
        half_life = config["half_life_minutes"]
        rules = config["rules"]

        if self.last_timestamp is None:
            self.fast_score = score
            self.sustained_score = score
            self.baseline_score = score
        else:
            elapsed = max(
                0.0,
                (timestamp - self.last_timestamp).total_seconds() / 60.0,
            )
            for field_name, key in [
                ("fast_score", "fast"),
                ("sustained_score", "sustained"),
                ("baseline_score", "baseline"),
            ]:
                previous = getattr(self, field_name)
                keep = retained_weight(elapsed, half_life[key])
                setattr(
                    self,
                    field_name,
                    keep * previous + (1.0 - keep) * score,
                )

        self.last_timestamp = timestamp
        self.post_count += 1
        self.recent_classes = (self.recent_classes + [clinical_class])[
            -rules["repeated_window_posts"]:
        ]

        immediate = (
            clinical_class == "Very High"
            and score >= rules["immediate_score"]
        )
        sustained = (
            self.post_count >= rules["sustained_minimum_posts"]
            and self.sustained_score >= rules["sustained_score"]
        )
        repeated = (
            len(self.recent_classes) == rules["repeated_window_posts"]
            and sum(
                value in {"High", "Very High"}
                for value in self.recent_classes
            )
            >= rules["repeated_minimum_high_or_very_high"]
        )
        escalation = (
            self.fast_score >= rules["escalation_minimum_fast_score"]
            and self.fast_score - self.baseline_score
            >= rules["escalation_fast_minus_baseline"]
        )

        raw_alert = immediate or sustained or repeated or escalation
        cooldown = timedelta(minutes=config["cooldown_minutes"])
        cooldown_active = (
            self.last_alert_timestamp is not None
            and timestamp - self.last_alert_timestamp < cooldown
        )
        alert = raw_alert and (immediate or not cooldown_active)
        if alert:
            self.last_alert_timestamp = timestamp

        return {
            "fast_score": self.fast_score,
            "sustained_score": self.sustained_score,
            "baseline_score": self.baseline_score,
            "immediate_alert": immediate,
            "sustained_alert": sustained,
            "repeated_alert": repeated,
            "escalation_alert": escalation,
            "cooldown_active": cooldown_active,
            "alert": alert,
        }


def load_default_config(project_root: str | Path):
    return json.loads(
        (Path(project_root) / "config/temporal_alert_defaults.json").read_text(
            encoding="utf-8"
        )
    )
