"""
Grant Trust System — Dashboard
================================
Full control gate dashboard: intent detection + fit scoring + reputation + pathways.

Local dev:
    pip install flask flask-cors numpy scipy scikit-learn spacy textstat langdetect
    python grant_trust_dashboard.py

Production:
    gunicorn grant_trust_dashboard:app
"""

import glob as globmod
import os, sys, json, time, threading, traceback
from typing import Dict, List

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from intent_engine import IntentDetectionEngine, ApplicantCorpus, FunderContext
from fit_scoring import FitScorer, FunderProfile
from reputation import ReputationScorer, ApplicantRecord
from pathways import PathwayGenerator
from control_gate import ControlGate, GateResult

app   = Flask(__name__)
CORS(app)
GATE = ControlGate()

# ── Load funder profiles and applicant records from data/ ─────────────────────

def load_funder_profiles():
    profiles = {}
    for path in sorted(globmod.glob(os.path.join(os.path.dirname(__file__), "data", "funders", "*.json"))):
        fp = FunderProfile.from_json(path)
        profiles[fp.funder_id] = fp
    return profiles

def load_applicant_records():
    records = {}
    for path in sorted(globmod.glob(os.path.join(os.path.dirname(__file__), "data", "applicants", "*.json"))):
        rec = ApplicantRecord.from_json(path)
        records[rec.applicant_id] = rec
    return records

FUNDER_PROFILES = load_funder_profiles()
APPLICANT_RECORDS = load_applicant_records()
ALL_FUNDERS = list(FUNDER_PROFILES.values())

# ── Demo data ─────────────────────────────────────────────────────────────────

DEMO_CORPUS = {
    "applicant_a": [
        """I've been working on Xenarch for about six months now, and version 2 finally
cracked the training stability problem that was killing me in v1. The trick was
combining gradient clipping at 1.0 with KL annealing over the first three epochs.
I lost count of how many runs hit NaN at epoch 8 before I figured that out.
The Apollo 11 test is a nice sanity check — if your model can't rank the lander
as the top anomaly in LROC imagery, something is fundamentally wrong with your
scoring. The contextual metric is doing most of the work for circular features.""",
        """The latent space geometry of the VAE matters more than I initially thought.
With z=56 and proper KL regularisation, the model learns to cluster geological
textures that share formation mechanisms — not just visual similarity. A fresh
crater and a degraded crater end up in different parts of the space even though
they look similar at first glance. This is why the false positive rate on boulder
fields dropped so sharply between v1 and v2.""",
    ],
    "applicant_b": [],
    "applicant_c": [
        """I spent most of last year building a distributed sensor network for urban
air quality monitoring. The hard part wasn't the hardware — it was getting the
calibration to hold up across different weather conditions. I ended up training
a gradient-boosted model on the sensor drift patterns and got RMSE down to
2.3 µg/m³ across 47 deployment sites. The PMS5003 sensors are cheap but noisy;
the calibration pipeline is where the real work lives."""
    ],
}

DEMO_APPLICATIONS = {
    "applicant_a": """I am applying for the Deep Science Ventures grant to support the next
development phase of Xenarch, an unsupervised anomaly detection system for planetary
surface technosignatures that I have been building for the past year.

The core technical challenge in this domain is the "rare but natural" problem:
fresh craters and unusual geology trigger false positives in naive outlier detectors.
My solution was to adapt a Variational Autoencoder trained exclusively on natural
geology, combined with a five-metric scoring system (reconstruction error, latent
density, contextual analysis, gradient anomaly, edge regularity). The contextual
metric carries 30% of the weight specifically to handle circular spacecraft.

Version 2, finalised in February 2026, achieved Rank 1 detection of the Apollo 11
lunar module at 99.58% confidence using both LROC NAC (0.5m/pixel) and Chandrayaan-2
OHRC (0.25m/pixel) imagery. The stability improvements — gradient clipping at 1.0,
KL annealing over 3 epochs, batch norm epsilon raised to 1e-3 — eliminated the NaN
divergence that plagued earlier versions. The code is open-source at github.com/calebstrom/xenarch.

The grant would fund cloud migration to NASA PDS Cloud for a global survey covering
Ceres, Europa, and Enceladus. I estimate covering the complete lunar surface at LROC
resolution in approximately 18 hours of compute time.""",

    "applicant_b_spray": """I am writing to express my deep interest in the Deep Science Ventures
grant opportunity. I am a passionate researcher with a strong background in artificial
intelligence and machine learning, and I believe my work would be a transformative
addition to your portfolio.

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
advance the boundaries of human knowledge. Thank you for considering my application.""",

    "applicant_b_v2": """I am writing to express my deep interest in the Science Foundation
grant opportunity. I am a passionate researcher with a strong background in artificial
intelligence and machine learning, and I believe my work would be a transformative
addition to your portfolio.

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
advance the boundaries of human knowledge. Thank you for considering my application.""",

    "applicant_c": """This proposal seeks funding for a community-based air quality monitoring
initiative using a distributed low-cost sensor network across three urban neighbourhoods.

Over the past 18 months, I have deployed 47 sensor nodes and developed a calibration
pipeline that maintains RMSE below 2.5 µg/m³ even under varying humidity conditions.
The sensors are based on the PMS5003 particulate matter sensor, cross-calibrated against
EPA reference monitors at three co-location sites. Data is published as open JSON via a
public API at airwatch-api.io.

I am applying to Deep Science Ventures because I believe environmental monitoring
infrastructure is an underserved area of frontier science. However, I recognise that
my work sits at the applied end of the spectrum — the innovation is in the calibration
methodology and deployment logistics rather than the underlying sensor physics.

The grant would fund expansion to 120 nodes and real-time public data access. All code
is open-source at github.com/[redacted]/airwatch.""",

    "applicant_b_improved": """I'm applying for the Deep Science Ventures grant to continue
development of AstroFind, an anomaly detection tool for planetary surface imagery that I
started building at SpaceHack 2026 two months ago.

The project uses a convolutional autoencoder to flag regions of Mars HiRISE imagery that
diverge statistically from normal terrain. My current model achieves a false positive rate
of around 12% on a holdout set of 200 labeled chips, which I know needs to improve. The
main issue is distinguishing fresh craters from genuinely anomalous compact features —
I've been reading Strom (2026) on multi-metric scoring and think the contextual metric
approach would help.

I shipped the first version of the pipeline in three weeks and presented it at the
hackathon. The code is on GitHub and three people have already opened issues, which has
been useful for identifying edge cases I'd missed. I'm looking for funding to spend two
months improving the scoring system and running it against the full HiRISE archive.""",
}


# ── Routes ────────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/api/status")
def api_status():
    return jsonify({"ready": True})

@app.route("/api/demo_data")
def api_demo_data():
    return jsonify({
        "applications": DEMO_APPLICATIONS,
        "corpus": DEMO_CORPUS,
        "applicants": {
            aid: {
                "name": rec.name,
                "hackathons": len(rec.hackathons),
                "open_source": len(rec.open_source),
                "publications": len(rec.publications),
                "collaborations": len(rec.collaborations),
            }
            for aid, rec in APPLICANT_RECORDS.items()
        },
        "funders": {
            fid: {
                "name": fp.name,
                "focus_areas": fp.focus_areas,
                "grant_type": fp.grant_type,
            }
            for fid, fp in FUNDER_PROFILES.items()
        },
    })

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True) or {}
    app_text = data.get("application_text", "").strip()
    if not app_text:
        return jsonify({"error": "application_text is required"}), 400

    funder_id = data.get("funder_id", "deep_science_ventures")
    applicant_id = data.get("applicant_id")
    corpus_txts = data.get("corpus", [])
    other_apps = data.get("other_applications", [])

    funder_profile = FUNDER_PROFILES.get(funder_id)
    if not funder_profile:
        return jsonify({"error": f"Unknown funder: {funder_id}"}), 400

    # Build or load applicant record
    if applicant_id and applicant_id in APPLICANT_RECORDS:
        applicant_record = APPLICANT_RECORDS[applicant_id]
    else:
        applicant_record = ApplicantRecord(
            applicant_id=applicant_id or "custom",
            name=data.get("applicant_name", "Custom Applicant"),
            corpus_texts=corpus_txts,
        )

    try:
        result = GATE.evaluate(
            application_text=app_text,
            applicant_record=applicant_record,
            funder_profile=funder_profile,
            other_applications=other_apps or None,
            all_funders=ALL_FUNDERS,
        )

        response = {
            # Gate decision
            "decision": result.decision,
            "match_quality": round(result.match_quality, 4),

            # Component 1: Intent
            "intent_score": result.intent_result.intent_score,
            "label": result.intent_result.label,
            "confidence": result.intent_result.confidence,
            "authorship_score": result.intent_result.authorship_score,
            "template_score": result.intent_result.template_score,
            "spray_score": result.intent_result.spray_score,
            "specificity_score": result.intent_result.specificity_score,
            "flags": result.intent_result.flags,
            "suggestions": result.intent_result.suggestions,

            # Component 2: Fit + Reputation
            "fit_score": round(result.fit_score, 4),
            "fit_detail": result.fit_detail,
            "reputation_score": round(result.reputation_score, 4),
            "reputation_detail": result.reputation_detail,

            # Component 3: Pathway (if failed)
            "pathway": None,
            "interrogation": None,
        }

        if result.pathway:
            response["pathway"] = {
                "primary_gap": result.pathway.primary_gap,
                "estimated_timeline": result.pathway.estimated_timeline,
                "resubmission_guidance": result.pathway.resubmission_guidance,
                "alternative_funders": result.pathway.alternative_funders,
                "actions": [
                    {
                        "category": a.category,
                        "action": a.action,
                        "expected_impact": a.expected_impact,
                        "difficulty": a.difficulty,
                        "time_estimate": a.time_estimate,
                    }
                    for a in result.pathway.actions
                ],
            }

        if result.interrogation and result.interrogation.triggered:
            response["interrogation"] = {
                "questions": result.interrogation.questions,
                "rationale": result.interrogation.rationale,
            }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5002))
    host  = os.environ.get("HOST", "0.0.0.0")
    print("=" * 60)
    print("GRANT TRUST SYSTEM — Control Gate Dashboard")
    print(f"  Gate       : {GATE.__class__.__name__}")
    print(f"  Funders    : {list(FUNDER_PROFILES.keys())}")
    print(f"  Applicants : {list(APPLICANT_RECORDS.keys())}")
    print(f"  Events     : {len(GATE.events)} loaded")
    print(f"  Presets    : {list(DEMO_APPLICATIONS.keys())}")
    print(f"  Serving    : http://{host}:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False, threaded=True)
