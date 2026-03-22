"""
Grant Trust System — Reputation Ledger
========================================
Component 2b: Costly signal layer

Scores an applicant's verifiable track record. This is the layer
an agent can't fake at scale — shipped projects, real commits,
published work.

See CLAUDE.md for full specification.

Owner: Shon Pan
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass
class HackathonEntry:
    name: str
    date: str                     # ISO format
    project: str
    outcome: str = "participated"  # "shipped" | "presented" | "participated"
    verifiable_url: Optional[str] = None


@dataclass
class OpenSourceEntry:
    repo: str
    stars: int = 0
    commits_last_year: int = 0
    description: str = ""
    verifiable_url: Optional[str] = None


@dataclass
class PublicationEntry:
    title: str
    venue: str
    year: int
    url: Optional[str] = None
    peer_reviewed: bool = False
    citations: int = 0


@dataclass
class CollaborationEntry:
    with_whom: str
    project: str
    collaboration_type: str = "team"   # "co_author" | "hackathon_team" | "institutional"
    date: Optional[str] = None
    verifiable_url: Optional[str] = None


@dataclass
class ApplicantRecord:
    """
    Full applicant record loaded from data/applicants/*.json.
    Contains both corpus (for Component 1) and reputation (for this module).
    """
    applicant_id: str
    name: str
    corpus_texts: List[str] = field(default_factory=list)
    hackathons: List[HackathonEntry] = field(default_factory=list)
    open_source: List[OpenSourceEntry] = field(default_factory=list)
    publications: List[PublicationEntry] = field(default_factory=list)
    collaborations: List[CollaborationEntry] = field(default_factory=list)
    previous_applications: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "ApplicantRecord":
        with open(path) as f:
            data = json.load(f)

        record = cls(
            applicant_id=data["applicant_id"],
            name=data.get("name", data["applicant_id"]),
            corpus_texts=data.get("corpus", {}).get("texts", []),
            previous_applications=data.get("previous_applications", []),
        )

        for h in data.get("reputation", {}).get("hackathons", []):
            record.hackathons.append(HackathonEntry(**h))
        for o in data.get("reputation", {}).get("open_source", []):
            record.open_source.append(OpenSourceEntry(**o))
        for p in data.get("reputation", {}).get("publications", []):
            record.publications.append(PublicationEntry(**p))
        for c in data.get("reputation", {}).get("collaborations", []):
            record.collaborations.append(CollaborationEntry(
                with_whom=c.get("with", c.get("with_whom", "")),
                project=c.get("project", ""),
                collaboration_type=c.get("type", c.get("collaboration_type", "team")),
                date=c.get("date"),
                verifiable_url=c.get("verifiable_url"),
            ))

        return record


class ReputationScorer:
    """
    Scores an applicant's verifiable track record.

    Scoring signals (from CLAUDE.md):
      - Hackathon participations: 0.30
      - Open source contributions: 0.25
      - Published work:           0.25
      - Verified collaborations:  0.20

    Rules:
      - Each category scores 0–1 independently
      - Recency decay: >2 years old = 50% discount
      - Verification bonus: entries with URLs score 20% higher
      - Empty ledger defaults to 0.15

    TODO: Implement scoring logic.
    """

    WEIGHTS = {
        "hackathons":     0.30,
        "open_source":    0.25,
        "publications":   0.25,
        "collaborations": 0.20,
    }

    EMPTY_LEDGER_DEFAULT = 0.15
    RECENCY_CUTOFF_DAYS = 730        # 2 years
    RECENCY_DISCOUNT = 0.5
    VERIFICATION_BONUS = 1.2

    def score(self, record: ApplicantRecord) -> Tuple[float, Dict]:
        """
        Returns (reputation_score, detail_dict).
        reputation_score: 0–1, where 1 = strong verifiable track record.
        """
        # Check for empty ledger
        has_any = (
            record.hackathons or record.open_source
            or record.publications or record.collaborations
        )
        if not has_any:
            return self.EMPTY_LEDGER_DEFAULT, {
                "hackathons": 0.0, "open_source": 0.0,
                "publications": 0.0, "collaborations": 0.0,
                "empty_ledger": True,
            }

        now = datetime.now()
        category_scores = {}

        # --- Hackathons (0.30) ---
        outcome_weights = {"shipped": 1.0, "presented": 0.6, "participated": 0.3}
        if record.hackathons:
            entry_scores = []
            for h in record.hackathons:
                base = outcome_weights.get(h.outcome, 0.3)
                base = self._apply_recency(base, h.date, now)
                if h.verifiable_url:
                    base *= self.VERIFICATION_BONUS
                entry_scores.append(min(1.0, base))
            category_scores["hackathons"] = sum(entry_scores) / len(entry_scores)
        else:
            category_scores["hackathons"] = 0.0

        # --- Open source (0.25) ---
        if record.open_source:
            entry_scores = []
            for o in record.open_source:
                star_component = min(1.0, o.stars / 50) * 0.4
                commit_component = min(1.0, o.commits_last_year / 100) * 0.6
                base = star_component + commit_component
                if o.verifiable_url:
                    base *= self.VERIFICATION_BONUS
                entry_scores.append(min(1.0, base))
            category_scores["open_source"] = sum(entry_scores) / len(entry_scores)
        else:
            category_scores["open_source"] = 0.0

        # --- Publications (0.25) ---
        if record.publications:
            entry_scores = []
            for p in record.publications:
                base = 0.7 if p.peer_reviewed else 0.4
                base += min(0.3, p.citations / 20)
                base = self._apply_recency(base, str(p.year), now)
                if p.url:
                    base *= self.VERIFICATION_BONUS
                entry_scores.append(min(1.0, base))
            category_scores["publications"] = sum(entry_scores) / len(entry_scores)
        else:
            category_scores["publications"] = 0.0

        # --- Collaborations (0.20) ---
        type_weights = {"co_author": 1.0, "institutional": 0.8, "hackathon_team": 0.6}
        if record.collaborations:
            entry_scores = []
            for c in record.collaborations:
                base = type_weights.get(c.collaboration_type, 0.5)
                base = self._apply_recency(base, c.date, now)
                if c.verifiable_url:
                    base *= self.VERIFICATION_BONUS
                entry_scores.append(min(1.0, base))
            category_scores["collaborations"] = sum(entry_scores) / len(entry_scores)
        else:
            category_scores["collaborations"] = 0.0

        # Weighted sum
        total = sum(
            category_scores[cat] * self.WEIGHTS[cat]
            for cat in self.WEIGHTS
        )
        total = min(1.0, total)

        return total, {**category_scores, "empty_ledger": False}

    def _apply_recency(self, score: float, date_str: Optional[str], now: datetime) -> float:
        """Apply recency decay: entries older than RECENCY_CUTOFF_DAYS get discounted."""
        if not date_str:
            return score
        try:
            # Handle both "YYYY-MM-DD" and "YYYY-MM" and "YYYY"
            if len(date_str) == 4:
                dt = datetime(int(date_str), 6, 15)
            elif len(date_str) <= 7:
                parts = date_str.split("-")
                dt = datetime(int(parts[0]), int(parts[1]), 15)
            else:
                dt = datetime.fromisoformat(date_str)
            age_days = (now - dt).days
            if age_days > self.RECENCY_CUTOFF_DAYS:
                score *= self.RECENCY_DISCOUNT
        except (ValueError, TypeError):
            pass
        return score
