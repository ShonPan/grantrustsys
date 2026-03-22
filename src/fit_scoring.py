"""
Grant Trust System — Fit Scoring
=================================
Component 2a: Funder profile matching

Compares an application against a funder profile to produce a fit score.
See CLAUDE.md for full specification.

Owner: Shon Pan
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import the FunderContext dataclass from Component 1
from intent_engine import FunderContext


@dataclass
class FunderProfile:
    """
    Extended funder profile loaded from data/funders/*.json.
    Wraps the FunderContext used by Component 1 with additional fields.
    """
    funder_id: str
    name: str
    mission_statement: str
    mission_keywords: List[str]
    focus_areas: List[str]
    past_recipient_language: List[str] = field(default_factory=list)
    budget_range: Dict = field(default_factory=lambda: {"min_usd": 0, "max_usd": 1_000_000})
    grant_type: str = "research"          # research | applied | infrastructure | fellowship
    application_cycle: str = "rolling"
    past_recipients: List[Dict] = field(default_factory=list)

    def to_funder_context(self) -> FunderContext:
        """Convert to the FunderContext dataclass used by Component 1."""
        return FunderContext(
            funder_id=self.funder_id,
            mission_keywords=self.mission_keywords,
            focus_areas=self.focus_areas,
            past_recipient_language=self.past_recipient_language,
        )

    @classmethod
    def from_json(cls, path: str) -> "FunderProfile":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class FitScorer:
    """
    Scores how well an application matches a funder's mission and scope.

    Scoring signals (from CLAUDE.md):
      - Mission semantic similarity: 0.40
      - Scope/budget alignment:     0.25
      - Domain match:               0.20
      - Past recipient language:    0.15

    TODO: Implement all four scoring signals.
    """

    def score(self,
              application_text: str,
              funder: FunderProfile,
              requested_budget: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Returns (fit_score, detail_dict).
        fit_score: 0–1, where 1 = perfect funder match.
        """
        text_lower = application_text.lower()
        text_words = set(re.findall(r'[a-z]{3,}', text_lower))
        detail = {}

        # --- Mission semantic similarity (0.40) ---
        # Keyword hit ratio with partial word matching (weight 0.5)
        keywords = funder.mission_keywords
        if keywords:
            hits = 0.0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in text_lower:
                    hits += 1.0  # exact phrase match
                else:
                    # partial: check individual words (>3 chars) from the keyword
                    kw_words = [w for w in re.findall(r'[a-z]{3,}', kw_lower)]
                    if kw_words:
                        word_hits = sum(1 for w in kw_words if w in text_words)
                        hits += 0.5 * (word_hits / len(kw_words))
            keyword_ratio = hits / len(keywords)
        else:
            keyword_ratio = 0.0

        # Bag-of-words cosine similarity (weight 0.5)
        from sklearn.feature_extraction.text import CountVectorizer
        from scipy.spatial.distance import cosine as cosine_dist

        funder_text = " ".join([
            funder.mission_statement,
            " ".join(funder.mission_keywords),
            " ".join(funder.focus_areas),
        ])
        try:
            vec = CountVectorizer(stop_words="english")
            matrix = vec.fit_transform([application_text, funder_text])
            a, b = matrix.toarray()[0], matrix.toarray()[1]
            bow_sim = 1.0 - cosine_dist(a, b) if np.any(a) and np.any(b) else 0.0
        except Exception:
            bow_sim = 0.0

        mission_score = 0.5 * keyword_ratio + 0.5 * bow_sim
        detail["mission_similarity"] = round(mission_score, 4)
        detail["keyword_ratio"] = round(keyword_ratio, 4)
        detail["bow_cosine"] = round(bow_sim, 4)

        # --- Scope/budget alignment (0.25) ---
        budget_min = funder.budget_range.get("min_usd", 0)
        budget_max = funder.budget_range.get("max_usd", 1_000_000)

        if requested_budget is None:
            budget_score = 0.5  # neutral
        elif budget_min <= requested_budget <= budget_max:
            budget_score = 1.0
        else:
            if requested_budget < budget_min:
                dist = (budget_min - requested_budget) / budget_min
            else:
                dist = (requested_budget - budget_max) / budget_max
            budget_score = max(0.0, 1.0 - dist)

        # Grant type keyword check
        grant_type_keywords = {
            "research": ["research", "methodology", "hypothesis", "analysis", "study", "validation"],
            "applied": ["deploy", "implement", "pilot", "prototype", "field", "sensor", "monitoring"],
            "infrastructure": ["infrastructure", "platform", "API", "pipeline", "tool"],
            "fellowship": ["career", "training", "mentorship", "development"],
        }
        gt_kws = grant_type_keywords.get(funder.grant_type, [])
        if gt_kws:
            gt_hits = sum(1 for kw in gt_kws if kw.lower() in text_lower)
            gt_ratio = min(1.0, gt_hits / max(1, len(gt_kws) * 0.4))
            budget_score = 0.6 * budget_score + 0.4 * gt_ratio
        detail["scope_budget"] = round(budget_score, 4)

        # --- Domain match (0.20) ---
        focus = funder.focus_areas
        if focus:
            domain_hits = 0.0
            for area in focus:
                area_lower = area.lower()
                if area_lower in text_lower:
                    domain_hits += 1.0
                else:
                    area_words = [w for w in re.findall(r'[a-z]{3,}', area_lower)]
                    if area_words:
                        word_hits = sum(1 for w in area_words if w in text_words)
                        domain_hits += 0.5 * (word_hits / len(area_words))
            domain_score = min(1.0, domain_hits / max(1, len(focus) * 0.4))
        else:
            domain_score = 0.0
        detail["domain_match"] = round(domain_score, 4)

        # --- Past recipient language (0.15) ---
        prl = funder.past_recipient_language
        if prl:
            prl_hits = 0.0
            for phrase in prl:
                phrase_lower = phrase.lower()
                if phrase_lower in text_lower:
                    prl_hits += 1.0
                else:
                    phrase_words = [w for w in re.findall(r'[a-z]{3,}', phrase_lower)]
                    if phrase_words:
                        word_hits = sum(1 for w in phrase_words if w in text_words)
                        prl_hits += 0.5 * (word_hits / len(phrase_words))
            prl_score = min(1.0, prl_hits / max(1, len(prl) * 0.4))
        else:
            prl_score = 0.0
        detail["past_recipient_language"] = round(prl_score, 4)

        # Weighted sum
        fit_score = (
            0.40 * mission_score
            + 0.25 * budget_score
            + 0.20 * domain_score
            + 0.15 * prl_score
        )
        fit_score = min(1.0, max(0.0, fit_score))

        return fit_score, detail
