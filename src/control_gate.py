"""
Grant Trust System — Control Gate
===================================
Integration layer: combines all three components into a single decision.

Match Quality = 0.35 × Intent + 0.40 × Fit + 0.25 × Reputation

≥ 0.65 → PASS
< 0.65 → FAIL → generate pathway
Intent in 0.45–0.60 → INTERROGATE (regardless of match quality)

See CLAUDE.md for full specification.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from intent_engine import IntentDetectionEngine, IntentScore, ApplicantCorpus, FunderContext
from fit_scoring import FitScorer, FunderProfile
from reputation import ReputationScorer, ApplicantRecord
from pathways import PathwayGenerator, ImprovementPathway, load_events


@dataclass
class InterrogationProtocol:
    """
    Stub for the interrogation protocol.
    When intent is in the mixed range (0.45–0.60), the trusted model
    (funder's review system) asks the applicant targeted factual questions
    about specific claims in the application.
    """
    triggered: bool = False
    questions: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class GateResult:
    """Full output of the control gate evaluation."""
    decision: str                                   # "PASS" | "INTERROGATE" | "FAIL"
    match_quality: float
    intent_result: IntentScore
    fit_score: float
    fit_detail: Dict
    reputation_score: float
    reputation_detail: Dict
    pathway: Optional[ImprovementPathway] = None
    interrogation: Optional[InterrogationProtocol] = None


class ControlGate:
    """
    Top-level entry point for the Grant Trust System.

    Usage:
        gate = ControlGate()
        result = gate.evaluate(application_text, applicant_record, funder_profile)
        print(result.decision, result.match_quality)

    TODO: Wire up all components and implement evaluate().
    """

    WEIGHTS = {
        "intent":     0.35,
        "fit":        0.40,
        "reputation": 0.25,
    }
    PASS_THRESHOLD = 0.65
    INTERROGATION_RANGE = (0.45, 0.60)

    def __init__(self, events_dir: str = None):
        self.intent_engine = IntentDetectionEngine()
        self.fit_scorer = FitScorer()
        self.reputation_scorer = ReputationScorer()
        self.pathway_generator = PathwayGenerator()
        if events_dir is None:
            events_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "events")
        self.events = load_events(events_dir)

    def evaluate(self,
                 application_text: str,
                 applicant_record: ApplicantRecord,
                 funder_profile: FunderProfile,
                 other_applications: Optional[List[str]] = None,
                 all_funders: Optional[List[FunderProfile]] = None) -> GateResult:
        """
        Run the full control gate evaluation.

        1. Run intent engine (Component 1)
        2. Run fit scoring (Component 2a)
        3. Run reputation scoring (Component 2b)
        4. Compute match quality
        5. Apply gate threshold
        6. If failed → generate improvement pathway (Component 3)
        7. If mixed intent → generate interrogation questions

        """
        # 1. Build Component 1 inputs
        corpus = ApplicantCorpus(
            applicant_id=applicant_record.applicant_id,
            texts=applicant_record.corpus_texts,
        )
        funder_context = funder_profile.to_funder_context()

        # 2. Run intent engine (Component 1)
        intent_result = self.intent_engine.analyze(
            application_text=application_text,
            corpus=corpus,
            other_applications=other_applications,
            funder_context=funder_context,
        )

        # 3. Run fit scoring (Component 2a)
        fit_score, fit_detail = self.fit_scorer.score(
            application_text=application_text,
            funder=funder_profile,
        )

        # 4. Run reputation scoring (Component 2b)
        rep_score, rep_detail = self.reputation_scorer.score(applicant_record)

        # 5. Compute match quality
        match_quality = (
            self.WEIGHTS["intent"] * intent_result.intent_score
            + self.WEIGHTS["fit"] * fit_score
            + self.WEIGHTS["reputation"] * rep_score
        )

        # 6. Decision logic
        intent_val = intent_result.intent_score
        low, high = self.INTERROGATION_RANGE

        interrogation = None
        pathway = None

        # Prepare pathway inputs
        funder_dicts = None
        if all_funders:
            funder_dicts = [
                {"funder_id": fp.funder_id, "name": fp.name,
                 "mission_keywords": fp.mission_keywords, "focus_areas": fp.focus_areas}
                for fp in all_funders
            ]
        applicant_dict = {
            "applicant_id": applicant_record.applicant_id,
            "corpus": {"texts": applicant_record.corpus_texts},
        }

        def _build_pathway():
            return self.pathway_generator.generate(
                intent_score=intent_val,
                intent_label=intent_result.label,
                fit_score=fit_score,
                reputation_score=rep_score,
                match_quality=match_quality,
                intent_suggestions=intent_result.suggestions,
                funder_profile={"funder_id": funder_profile.funder_id, "name": funder_profile.name},
                applicant_record=applicant_dict,
                all_funders=funder_dicts,
                events=self.events,
            )

        if low <= intent_val <= high:
            decision = "INTERROGATE"
            interrogation = self._generate_interrogation(application_text, intent_result)
            if match_quality < self.PASS_THRESHOLD:
                pathway = _build_pathway()
        elif match_quality >= self.PASS_THRESHOLD:
            decision = "PASS"
        else:
            decision = "FAIL"
            pathway = _build_pathway()

        return GateResult(
            decision=decision,
            match_quality=round(match_quality, 4),
            intent_result=intent_result,
            fit_score=round(fit_score, 4),
            fit_detail=fit_detail,
            reputation_score=round(rep_score, 4),
            reputation_detail=rep_detail,
            pathway=pathway,
            interrogation=interrogation,
        )

    def _generate_interrogation(self,
                                application_text: str,
                                intent_result: IntentScore) -> InterrogationProtocol:
        """
        Generate targeted factual questions about specific claims
        in the application. These are questions only the real researcher
        could answer — the equivalent of a CAPTCHA for expertise.

        Example questions:
          "You mention achieving RMSE of 2.3 µg/m³ — what was the baseline
           before your calibration pipeline?"
          "You reference 47 deployment sites — which neighbourhood had the
           highest variance and why?"

        """
        questions = []
        # Normalize whitespace for regex matching
        text = " ".join(application_text.split())

        # Find numeric claims (percentages, measurements, counts)
        numeric_claims = re.findall(
            r'(?:achieved|reaching|attained|got|maintains?|below|under|above|approximately|about)\s+[^.]*?(?:\d+[\d.,]*\s*(?:%|percent|µg|mg|kg|m³|sites?|nodes?|hours?|epochs?|pixels?))',
            text, re.IGNORECASE
        )
        for claim in numeric_claims[:2]:
            claim = claim.strip()
            questions.append(
                f"You mention '{claim}' — what was the baseline before this result, and how did you validate it?"
            )

        # Find specific counts
        count_matches = re.findall(r'(\d+)\s+(sensor nodes?|deployment sites?|nodes|sites|test cases?|application pairs?)', text, re.IGNORECASE)
        for count, thing in count_matches[:1]:
            questions.append(
                f"You reference {count} {thing} — which specific location or case showed the most variance, and why?"
            )

        # Find technical method claims
        method_matches = re.findall(
            r'(?:using|based on|built|developed|trained|pipeline|approach|framework)\s+(?:a\s+)?([^.]{10,60}?)(?:\.|,|that|which)',
            text, re.IGNORECASE
        )
        for method in method_matches[:1]:
            questions.append(
                f"You describe '{method.strip()}' — what alternatives did you consider, and why did you choose this approach?"
            )

        # Find project/tool names
        project_names = re.findall(r'(?:called|named|dubbed)\s+(\w+)|(\w+(?:Scan|Watch|Arch|Net|Lab))', text)
        for match in project_names[:1]:
            name = match[0] or match[1]
            questions.append(
                f"Can you walk through a specific debugging session you had while building {name}?"
            )

        # Ensure at least 3 questions
        if len(questions) < 3:
            questions.append(
                "Describe a specific technical obstacle you encountered in this work and how you resolved it."
            )
        if len(questions) < 3:
            questions.append(
                "What was the most surprising finding in your results, and how did it change your approach?"
            )

        return InterrogationProtocol(
            triggered=True,
            questions=questions[:5],
            rationale=f"Intent score {intent_result.intent_score:.2f} falls in mixed range ({self.INTERROGATION_RANGE[0]}–{self.INTERROGATION_RANGE[1]}). "
                      f"Generating targeted factual questions to verify authentic engagement.",
        )
