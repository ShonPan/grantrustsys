"""
Grant Trust System — Full Demo
================================
Runs the four cases from the whitepaper:
  Case 1: Applicant A — Genuine researcher, AI-assisted → PASS
  Case 2: Applicant B — Spray-and-pray, no corpus → FAIL + pathway
  Case 3: Applicant C — Genuine, wrong funder → intent PASS, fit FAIL + redirect
  Case 4: Applicant B v2 — Returns after pathway actions → INTERROGATE

Loads all data from data/ directory.

Usage:
    cd grant-trust-system
    python demo.py
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from intent_engine import (
    IntentDetectionEngine, ApplicantCorpus, FunderContext
)

from fit_scoring import FitScorer, FunderProfile
from reputation import ReputationScorer, ApplicantRecord
from control_gate import ControlGate, GateResult


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_funders():
    """Load all funder profiles from data/funders/."""
    funders = {}
    funder_dir = os.path.join(DATA_DIR, "funders")
    for fname in os.listdir(funder_dir):
        if fname.endswith(".json"):
            with open(os.path.join(funder_dir, fname)) as f:
                data = json.load(f)
                funders[data["funder_id"]] = data
    return funders


def load_applicants():
    """Load all applicant records from data/applicants/."""
    applicants = {}
    app_dir = os.path.join(DATA_DIR, "applicants")
    for fname in os.listdir(app_dir):
        if fname.endswith(".json"):
            with open(os.path.join(app_dir, fname)) as f:
                data = json.load(f)
                applicants[data["applicant_id"]] = data
    return applicants


def make_corpus(applicant_data):
    """Convert applicant JSON to ApplicantCorpus for Component 1."""
    return ApplicantCorpus(
        applicant_id=applicant_data["applicant_id"],
        texts=applicant_data.get("corpus", {}).get("texts", []),
        metadata=applicant_data.get("corpus", {}).get("metadata", {}),
    )


def make_funder_context(funder_data):
    """Convert funder JSON to FunderContext for Component 1."""
    return FunderContext(
        funder_id=funder_data["funder_id"],
        mission_keywords=funder_data.get("mission_keywords", []),
        focus_areas=funder_data.get("focus_areas", []),
        past_recipient_language=funder_data.get("past_recipient_language", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION TEXTS (the grant applications being evaluated)
# ─────────────────────────────────────────────────────────────────────────────

APPLICATION_A = """
I am applying for the Deep Science Ventures grant to support the next development
phase of Xenarch, an unsupervised anomaly detection system for planetary surface
technosignatures that I have been building for the past year.

The core technical challenge in this domain is the "rare but natural" problem:
fresh craters and unusual geology trigger false positives in naive outlier detectors.
My solution was to adapt a Variational Autoencoder trained exclusively on natural
geology, combined with a five-metric scoring system (reconstruction error, latent
density, contextual analysis, gradient anomaly, edge regularity). The contextual
metric carries 30% of the weight specifically to handle circular spacecraft.

Version 2, which I finalised in February 2026, achieved Rank 1 detection of the
Apollo 11 lunar module at 99.58% confidence using both LROC NAC (0.5m/pixel) and
Chandrayaan-2 OHRC (0.25m/pixel) imagery. The training stability improvements —
gradient clipping at 1.0, KL annealing over the first 3 epochs, batch norm epsilon
raised to 1e-3 — eliminated the NaN divergence that plagued earlier versions.

The grant would fund cloud migration to NASA PDS Cloud for a global survey covering
Ceres, Europa, and Enceladus. Based on my validation results, I estimate covering
the complete lunar surface at LROC resolution in approximately 18 hours of compute
time. I have published the methodology at arxiv.org/abs/2026.xxxxx and the code is
open-source on GitHub at github.com/calebstrom/xenarch.
"""

APPLICATION_B_V1 = """
I am writing to express my deep interest in the Deep Science Ventures grant opportunity.
I am a passionate researcher with a strong background in artificial intelligence and
machine learning, and I believe my work would be a transformative addition to your
portfolio.

My research leverages cutting-edge deep learning techniques to drive innovation in
scientific discovery. I am deeply committed to fostering collaboration and creating
impactful outcomes that align with your organisation's holistic approach to funding
frontier science. I have a robust framework for conducting rigorous research and a
proven track record of delivering high-quality results.

Furthermore, my interdisciplinary background enables me to synthesize insights across
multiple domains and develop paradigm-shifting solutions to complex problems. I am
excited to bring my passion and expertise to this opportunity and look forward to
contributing to your mission of supporting transformative scientific breakthroughs.

In conclusion, I believe my skills and experience make me an ideal candidate for this
grant. I am committed to leveraging this funding to create meaningful impact and
advance the boundaries of human knowledge. Thank you for considering my application.
"""

APPLICATION_B_V2 = """
I am writing to express my deep interest in the Science Foundation grant opportunity.
I am a passionate researcher with a strong background in artificial intelligence and
machine learning, and I believe my work would be a transformative addition to your
portfolio.

My research leverages cutting-edge deep learning techniques to drive innovation in
scientific discovery. I am deeply committed to fostering collaboration and creating
impactful outcomes that align with your organisation's holistic approach to funding
frontier research. I have a robust framework for conducting rigorous research and a
proven track record of delivering high-quality results.

Furthermore, my interdisciplinary background enables me to synthesize insights across
multiple domains and develop paradigm-shifting solutions to complex problems. I am
excited to bring my passion and expertise to this opportunity and look forward to
contributing to your mission of supporting transformative scientific progress.

In conclusion, I believe my skills and experience make me an ideal candidate for this
grant. I am committed to leveraging this funding to create meaningful impact and
advance the boundaries of human knowledge. Thank you for considering my application.
"""

APPLICATION_C = """
This proposal seeks funding for a community-based air quality monitoring initiative
using a distributed low-cost sensor network across three urban neighbourhoods.

Over the past 18 months, I have deployed 47 sensor nodes and developed a calibration
pipeline that maintains RMSE below 2.5 µg/m³ even under varying humidity conditions.
The sensors are based on the PMS5003 particulate matter sensor, cross-calibrated
against EPA reference monitors at three co-location sites.

I am applying to Deep Science Ventures because I believe environmental monitoring
infrastructure is an underserved area of frontier science. However, I recognise that
my work sits at the applied end of the spectrum — the innovation is in the calibration
methodology and deployment logistics rather than the underlying sensor physics.

The grant would fund the expansion to 120 nodes and a public API for real-time data
access. All data and code are open-source at github.com/[redacted]/airwatch.
"""

# Case 4: Applicant B returns after intervention
APPLICATION_B_POST_INTERVENTION = """
I am applying for the Deep Science Ventures grant to support development of GrantScan,
a template-reuse detection tool for research funding applications.

At SpaceHack 2026 I built the first version with a team of three. The core approach
uses a sliding window over paragraph-level TF-IDF vectors to detect when structural
templates are reused across applications. We compared attention-weight distributions
between paragraphs and flagged cases where the structural fingerprint matched but
the content words were swapped — the signature of find-and-replace grant writing.

The current prototype catches 78% of template reuse with a false positive rate under
5%, tested against a hand-labelled set of 200 application pairs. The main failure mode
is legitimately shared methodology sections, which we handle by scoring structural
similarity and content similarity separately.

I recognise this is adjacent to rather than central to DSV's mission in frontier
science. But I think the integrity of the grant ecosystem is a precondition for
effective science funding, and GrantScan could serve as infrastructure for funders
who want to verify the authenticity of applications they receive. The code is at
github.com/asmith/grantscan.
"""


# ─────────────────────────────────────────────────────────────────────────────
# DEMO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def print_header(text):
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print(f"{'─'*70}")


def print_intent_result(result):
    """Pretty-print a Component 1 IntentScore."""
    print(f"  Intent score:      {result.intent_score:.4f}")
    print(f"  Label:             {result.label}")
    print(f"  Confidence:        {result.confidence:.4f}")
    print(f"\n  Component scores:")
    print(f"    Authorship:      {result.authorship_score:.4f}")
    print(f"    Template dist:   {result.template_score:.4f}")
    print(f"    Spray score:     {result.spray_score:.4f}  (lower = better)")
    print(f"    Specificity:     {result.specificity_score:.4f}")
    print(f"\n  Flags:")
    for flag in result.flags:
        print(f"    · {flag}")
    print(f"\n  Suggestions:")
    for s in result.suggestions:
        print(f"    → {s}")


def run_component_1_demo():
    """
    Run Component 1 (Intent Detection) standalone.
    This works now with Caleb's completed engine.
    """
    print("\n" + "=" * 70)
    print("GRANT TRUST SYSTEM — COMPONENT 1 DEMO (Intent Detection)")
    print("=" * 70)

    funders = load_funders()
    applicants = load_applicants()
    engine = IntentDetectionEngine()

    dsv = make_funder_context(funders["deep_science_ventures"])

    # Case 1: Genuine researcher
    print_header("CASE 1: Applicant A — Genuine researcher, AI-augmented")
    result = engine.analyze(
        application_text=APPLICATION_A,
        corpus=make_corpus(applicants["applicant_a"]),
        funder_context=dsv,
    )
    print_intent_result(result)

    # Case 2: Spray-and-pray
    print_header("CASE 2: Applicant B — Spray-and-pray (first application)")
    result = engine.analyze(
        application_text=APPLICATION_B_V1,
        corpus=make_corpus(applicants["applicant_b"]),
        other_applications=[APPLICATION_B_V2],
        funder_context=dsv,
    )
    print_intent_result(result)

    # Case 3: Genuine, wrong funder
    print_header("CASE 3: Applicant C — Real person, wrong funder (DSV)")
    result = engine.analyze(
        application_text=APPLICATION_C,
        corpus=make_corpus(applicants["applicant_c"]),
        funder_context=dsv,
    )
    print_intent_result(result)

    # Case 4: Applicant B post-intervention
    print_header("CASE 4: Applicant B — Post-intervention (after SpaceHack)")
    result = engine.analyze(
        application_text=APPLICATION_B_POST_INTERVENTION,
        corpus=make_corpus(applicants["applicant_b_v2"]),
        funder_context=dsv,
    )
    print_intent_result(result)

    print(f"\n{'='*70}")


def run_full_demo():
    """
    Run the full three-component demo with the control gate.

    TODO: Enable this once Components 2 & 3 are built.
    Uncomment the imports at the top of this file, then replace
    run_component_1_demo() with run_full_demo() in __main__.
    """
    print("\n" + "=" * 70)
    print("GRANT TRUST SYSTEM — FULL DEMO (All Components)")
    print("=" * 70)

    funders = load_funders()
    applicants = load_applicants()
    gate = ControlGate()

    # Load funder profiles
    dsv = FunderProfile.from_json(os.path.join(DATA_DIR, "funders", "deep_science_ventures.json"))
    ghf = FunderProfile.from_json(os.path.join(DATA_DIR, "funders", "green_horizons_fund.json"))
    nxb = FunderProfile.from_json(os.path.join(DATA_DIR, "funders", "nexgen_biotech.json"))
    raf = FunderProfile.from_json(os.path.join(DATA_DIR, "funders", "responsible_ai_foundation.json"))
    all_funders = [dsv, ghf, nxb, raf]

    # Case 1: Genuine researcher → expects PASS
    print_header("CASE 1: Applicant A — Genuine researcher → PASS")
    record_a = ApplicantRecord.from_json(os.path.join(DATA_DIR, "applicants", "applicant_a.json"))
    result_1 = gate.evaluate(APPLICATION_A, record_a, dsv, all_funders=all_funders)
    print_gate_result(result_1)

    # Case 2: Spray-and-pray → expects FAIL + pathway
    print_header("CASE 2: Applicant B — Spray-and-pray → FAIL")
    record_b = ApplicantRecord.from_json(os.path.join(DATA_DIR, "applicants", "applicant_b.json"))
    result_2 = gate.evaluate(APPLICATION_B_V1, record_b, dsv,
                             other_applications=[APPLICATION_B_V2], all_funders=all_funders)
    print_gate_result(result_2)

    # Case 3: Genuine, wrong funder → expects intent PASS, fit FAIL + redirect
    print_header("CASE 3: Applicant C — Wrong funder → REDIRECT")
    record_c = ApplicantRecord.from_json(os.path.join(DATA_DIR, "applicants", "applicant_c.json"))
    result_3 = gate.evaluate(APPLICATION_C, record_c, dsv, all_funders=all_funders)
    print_gate_result(result_3)

    # Case 4: Post-intervention → expects INTERROGATE
    print_header("CASE 4: Applicant B v2 — Post-intervention → INTERROGATE")
    record_b2 = ApplicantRecord.from_json(os.path.join(DATA_DIR, "applicants", "applicant_b_v2.json"))
    result_4 = gate.evaluate(APPLICATION_B_POST_INTERVENTION, record_b2, dsv, all_funders=all_funders)
    print_gate_result(result_4)


def print_gate_result(result: GateResult):
    print(f"  DECISION:          {result.decision}")
    print(f"  Match Quality:     {result.match_quality:.4f}")
    print(f"  Intent Score:      {result.intent_result.intent_score:.4f} ({result.intent_result.label})")
    print(f"  Fit Score:         {result.fit_score:.4f}")
    print(f"  Reputation Score:  {result.reputation_score:.4f}")
    if result.pathway:
        print(f"\n  Improvement Pathway:")
        print(f"    Primary gap:     {result.pathway.primary_gap}")
        print(f"    Timeline:        {result.pathway.estimated_timeline}")
        for action in result.pathway.actions:
            print(f"    -> [{action.category}] {action.action}")
        if result.pathway.alternative_funders:
            print(f"    Better-matched funders: {', '.join(result.pathway.alternative_funders)}")
    if result.interrogation and result.interrogation.triggered:
        print(f"\n  Interrogation Protocol (mixed intent detected):")
        for q in result.interrogation.questions:
            print(f"    ? {q}")


if __name__ == "__main__":
    run_full_demo()
