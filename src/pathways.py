"""
Grant Trust System — Improvement Pathways
===========================================
Component 3: Developmental feedback loop

When an applicant fails the control gate, generates structured,
actionable next steps. This is the mechanism that makes the protocol
developmental rather than purely restrictive — a ladder, not a wall.

See CLAUDE.md for full specification.

Owner: Shon Pan
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fit_scoring import FunderProfile
from reputation import ApplicantRecord


@dataclass
class PathwayAction:
    """A single recommended action in an improvement pathway."""
    category: str           # "writing" | "experience" | "funder_fit" | "reputation"
    action: str             # human-readable description of what to do
    expected_impact: str    # which score this improves and by roughly how much
    difficulty: str         # "easy" | "medium" | "hard"
    time_estimate: str      # "1 week" | "2 weeks" | "30 days" | "90 days"
    verifiable: bool        # can the result be added to the reputation ledger?


@dataclass
class ImprovementPathway:
    """Full improvement pathway for a rejected applicant."""
    primary_gap: str                     # "intent" | "fit" | "reputation" | "combined"
    actions: List[PathwayAction]         # ordered list of recommended actions
    estimated_timeline: str              # overall timeline to become competitive
    resubmission_guidance: str           # what would change on re-evaluation
    alternative_funders: List[str] = field(default_factory=list)  # if fit is the issue


class PathwayGenerator:
    """
    Generates improvement pathways based on the full score breakdown.

    Pathway logic (from whitepaper):
      - Low intent score → writing guidance
      - Low reputation score → experience building
      - Low fit score → funder redirection
      - Combined low → sequenced 90-day plan

    Completed pathway actions feed back into the reputation ledger.
    This closes the feedback loop.

    TODO: Implement pathway generation logic.
    """

    def generate(self,
                 intent_score: float,
                 intent_label: str,
                 fit_score: float,
                 reputation_score: float,
                 match_quality: float,
                 intent_suggestions: List[str],
                 funder_profile: Optional[dict] = None,
                 applicant_record: Optional[dict] = None,
                 all_funders: Optional[List[dict]] = None) -> ImprovementPathway:
        """
        Returns an ImprovementPathway with ordered actions.

        Parameters
        ----------
        intent_score        From Component 1
        intent_label        From Component 1 ("spray_and_pray", "likely_automated", etc.)
        fit_score           From Component 2a
        reputation_score    From Component 2b
        match_quality       From the control gate formula
        intent_suggestions  From Component 1's _generate_suggestions()
        funder_profile      Current funder (for redirection logic)
        applicant_record    Current applicant (for personalized recommendations)
        all_funders         All available funders (for alternative matching)
        """
        # Determine primary gap
        if intent_score < 0.45 and reputation_score < 0.40 and fit_score < 0.50:
            primary_gap = "combined"
        elif intent_score < 0.45:
            primary_gap = "intent"
        elif reputation_score < 0.40:
            primary_gap = "reputation"
        elif fit_score < 0.50:
            primary_gap = "fit"
        else:
            primary_gap = "combined"  # marginal case

        actions: List[PathwayAction] = []
        alternative_funders: List[str] = []

        # --- Intent gap actions ---
        if primary_gap in ("intent", "combined"):
            # Include Component 1's suggestions
            for suggestion in intent_suggestions:
                actions.append(PathwayAction(
                    category="writing",
                    action=suggestion,
                    expected_impact="Intent score +0.10–0.20",
                    difficulty="easy",
                    time_estimate="1 week",
                    verifiable=False,
                ))
            actions.append(PathwayAction(
                category="writing",
                action="Write a blog post or README explaining your project in your own voice before drafting the application",
                expected_impact="Intent score +0.15 (builds authentic corpus)",
                difficulty="easy",
                time_estimate="1 week",
                verifiable=True,
            ))

        # --- Reputation gap actions ---
        if primary_gap in ("reputation", "combined"):
            actions.append(PathwayAction(
                category="experience",
                action="Participate in a hackathon and ship a working prototype related to your research area",
                expected_impact="Reputation score +0.15–0.25",
                difficulty="medium",
                time_estimate="2 weeks",
                verifiable=True,
            ))
            actions.append(PathwayAction(
                category="experience",
                action="Open-source your code on GitHub with a clear README and at least 20 commits showing iterative development",
                expected_impact="Reputation score +0.10–0.15",
                difficulty="medium",
                time_estimate="30 days",
                verifiable=True,
            ))
            actions.append(PathwayAction(
                category="reputation",
                action="Publish a preprint or technical writeup documenting your methodology and results",
                expected_impact="Reputation score +0.10",
                difficulty="hard",
                time_estimate="30 days",
                verifiable=True,
            ))

        # --- Fit gap actions ---
        if primary_gap in ("fit", "combined"):
            actions.append(PathwayAction(
                category="funder_fit",
                action="Review the funder's mission statement and past recipients, then rewrite your application to connect your work to their specific focus areas",
                expected_impact="Fit score +0.15–0.25",
                difficulty="easy",
                time_estimate="1 week",
                verifiable=False,
            ))
            # Find alternative funders by comparing applicant's domain words
            # against each funder's keywords and focus areas
            if all_funders and funder_profile:
                import re as _re
                current_id = funder_profile.get("funder_id", "")
                # Build word set from applicant's corpus
                app_words = set()
                if applicant_record:
                    for text in applicant_record.get("corpus", {}).get("texts", []):
                        app_words.update(_re.findall(r'[a-z]{3,}', text.lower()))

                scored_funders = []
                for f in all_funders:
                    f_data = f if isinstance(f, dict) else vars(f)
                    if f_data.get("funder_id") == current_id:
                        continue
                    # Extract individual words from funder's keywords and areas
                    f_words = set()
                    for phrase in f_data.get("mission_keywords", []) + f_data.get("focus_areas", []):
                        f_words.update(_re.findall(r'[a-z]{3,}', phrase.lower()))
                    overlap = len(app_words & f_words)
                    if overlap >= 2:
                        scored_funders.append((overlap, f_data.get("name", f_data.get("funder_id", "Unknown"))))

                # Sort by overlap descending, take top 2
                scored_funders.sort(reverse=True)
                alternative_funders = [name for _, name in scored_funders[:2]]

            if alternative_funders:
                actions.append(PathwayAction(
                    category="funder_fit",
                    action=f"Consider applying to better-matched funders: {', '.join(alternative_funders)}",
                    expected_impact="Fit score +0.30+ with correct funder",
                    difficulty="easy",
                    time_estimate="1 week",
                    verifiable=False,
                ))

        # Determine timeline
        if primary_gap == "combined":
            estimated_timeline = "90 days"
        elif primary_gap == "reputation":
            estimated_timeline = "30 days"
        else:
            estimated_timeline = "2 weeks"

        # Resubmission guidance
        guidance_map = {
            "intent": "After building your corpus with authentic writing, your intent score should improve. Resubmit with a fresh application written in your own voice.",
            "fit": "After reviewing the funder's priorities and refocusing your application, resubmit. Or apply to a better-matched funder.",
            "reputation": "After completing the experience-building actions and adding verifiable entries to your record, resubmit.",
            "combined": "Complete the sequenced actions over 90 days: first build authentic writing, then gain verifiable experience, then resubmit with a focused application.",
        }
        resubmission_guidance = guidance_map.get(primary_gap, guidance_map["combined"])

        return ImprovementPathway(
            primary_gap=primary_gap,
            actions=actions,
            estimated_timeline=estimated_timeline,
            resubmission_guidance=resubmission_guidance,
            alternative_funders=alternative_funders,
        )
