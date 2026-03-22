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
from flask import Flask, request, jsonify, render_template_string
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

# ── Frontend HTML ─────────────────────────────────────────────────────────────

FRONTEND = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>GRANT TRUST SYSTEM · Control Gate</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
/* ── reset + vars ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#04060d;--surface:#090e1a;--panel:#0d1526;--border:#1a2744;
  --accent:#00e5ff;--accent2:#ff4f00;--green:#00d97e;--amber:#ffb800;--red:#ff4444;
  --dim:#3a5080;--text:#c8daf5;--textlo:#4a6080;
  --mono:'Share Tech Mono',monospace;--sans:'Barlow Condensed',sans-serif;
  --glow:0 0 18px rgba(0,229,255,.35);--glow-g:0 0 14px rgba(0,217,126,.4);
  --glow-r:0 0 14px rgba(255,68,68,.4);--glow-a:0 0 14px rgba(255,184,0,.4);
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:15px}
body::before{content:'';position:fixed;inset:0;z-index:9999;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px)}

/* ── layout ── */
.shell{max-width:1400px;margin:0 auto;padding:0 24px 80px}

/* ── header ── */
header{display:flex;align-items:center;gap:18px;padding:26px 0 20px;
  border-bottom:1px solid var(--border);margin-bottom:28px}
.logo-mark{width:44px;height:44px;border:2px solid var(--accent);border-radius:4px;
  display:grid;place-items:center;box-shadow:var(--glow);position:relative;flex-shrink:0}
.logo-mark::after{content:'';position:absolute;inset:5px;border:1px solid var(--accent);
  border-radius:2px;opacity:.5}
.logo-cross{width:16px;height:16px;
  background:linear-gradient(var(--accent),var(--accent)) 50% 0/2px 100%,
             linear-gradient(var(--accent),var(--accent)) 0 50%/100% 2px;
  background-color:transparent}
@keyframes scanY{0%,100%{transform:translateY(0);opacity:1}50%{transform:translateY(28px);opacity:.4}}
.logo-cross{animation:scanY 3s ease-in-out infinite}
.logo-text h1{font-family:var(--mono);font-size:19px;letter-spacing:.18em;color:var(--accent);text-shadow:var(--glow)}
.logo-text p{font-size:11px;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-top:2px}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:20px}
.track-badge{font-family:var(--mono);font-size:10px;letter-spacing:.12em;
  border:1px solid var(--border);border-radius:3px;padding:4px 10px;color:var(--dim)}
.track-badge span{color:var(--accent)}
.engine-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:var(--glow-g);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── two-col ── */
.main-grid{display:grid;grid-template-columns:380px 1fr;gap:22px;align-items:start}
@media(max-width:960px){.main-grid{grid-template-columns:1fr}}

/* ── panel ── */
.panel{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:20px;margin-bottom:18px}
.panel-title{font-family:var(--mono);font-size:10px;letter-spacing:.18em;color:var(--dim);
  text-transform:uppercase;margin-bottom:14px;padding-bottom:9px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px}
.panel-title-tag{font-size:9px;letter-spacing:.08em;padding:2px 7px;border-radius:2px;
  background:rgba(0,229,255,.12);color:var(--accent);border:1px solid rgba(0,229,255,.2)}

/* ── preset tabs ── */
.tab-row{display:flex;gap:0;border:1px solid var(--border);border-radius:4px;overflow:hidden;margin-bottom:14px}
.tab-btn{flex:1;padding:8px 3px;background:transparent;border:none;border-right:1px solid var(--border);
  color:var(--textlo);font-family:var(--mono);font-size:9px;letter-spacing:.06em;cursor:pointer;
  text-transform:uppercase;transition:background .15s,color .15s;line-height:1.4;text-align:center}
.tab-btn:last-child{border-right:none}
.tab-btn:hover{background:rgba(0,229,255,.06);color:var(--text)}
.tab-btn.active{background:rgba(0,229,255,.14);color:var(--accent)}
.tab-btn .tab-icon{display:block;font-size:14px;margin-bottom:2px}

/* ── form fields ── */
.field{margin-bottom:13px}
.field label{display:block;font-size:10px;letter-spacing:.12em;color:var(--dim);
  margin-bottom:5px;text-transform:uppercase}
.field select,.field textarea{width:100%;background:var(--surface);border:1px solid var(--border);
  border-radius:4px;color:var(--text);font-family:var(--mono);font-size:12px;padding:8px 10px;
  outline:none;transition:border-color .15s;resize:vertical}
.field select:focus,.field textarea:focus{border-color:var(--accent)}
.field textarea{min-height:110px;line-height:1.6}
.field .field-hint{font-size:10px;color:var(--textlo);margin-top:4px;letter-spacing:.04em}

/* ── analyse button ── */
#run-btn{width:100%;padding:13px;background:transparent;border:2px solid var(--accent);
  border-radius:4px;color:var(--accent);font-family:var(--mono);font-size:13px;
  letter-spacing:.18em;cursor:pointer;text-transform:uppercase;box-shadow:var(--glow);
  transition:background .2s,color .2s,box-shadow .2s;margin-top:4px}
#run-btn:hover:not(:disabled){background:var(--accent);color:var(--bg);box-shadow:0 0 28px rgba(0,229,255,.6)}
#run-btn:disabled{opacity:.35;cursor:not-allowed}

/* ── error ── */
.error-msg{background:rgba(255,68,68,.1);border:1px solid rgba(255,68,68,.35);border-radius:4px;
  padding:10px 14px;font-family:var(--mono);font-size:11px;color:#ff8a8a;margin-top:10px;display:none}

/* ── right column sections ── */
.right-col{display:flex;flex-direction:column}

/* ── loading ── */
#loading{display:none;text-align:center;padding:56px 0}
.spinner{width:32px;height:32px;border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 14px}
@keyframes spin{to{transform:rotate(360deg)}}
#loading p{font-family:var(--mono);font-size:11px;color:var(--dim);letter-spacing:.1em}

/* ── result panel ── */
#result-panel{display:none}

/* ── verdict hero ── */
.verdict{display:flex;align-items:center;gap:24px;padding:20px;
  background:var(--surface);border:1px solid var(--border);border-radius:6px;margin-bottom:18px}
.ring-wrap{position:relative;width:104px;height:104px;flex-shrink:0}
.ring-wrap svg{width:104px;height:104px}
.ring-center{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center}
.ring-num{font-family:var(--mono);font-size:26px;font-weight:700;line-height:1}
.ring-unit{font-size:9px;color:var(--dim);letter-spacing:.1em;margin-top:2px;text-transform:uppercase}
.verdict-meta{flex:1;min-width:0}
.label-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;
  border-radius:3px;font-family:var(--mono);font-size:11px;letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:10px}
.label-pill-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.verdict-desc{font-size:14px;color:var(--textlo);line-height:1.6;margin-bottom:10px}
.verdict-conf{font-family:var(--mono);font-size:11px;color:var(--dim)}
.verdict-conf span{color:var(--text)}

/* ── component breakdown ── */
.breakdown{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:14px}
.mc-top{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:8px}
.mc-label{font-size:10px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase}
.mc-val{font-family:var(--mono);font-size:22px;font-weight:700;line-height:1}
.mc-bar-outer{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:7px}
.mc-bar-inner{height:100%;border-radius:2px;transition:width .7s ease}
.mc-desc{font-size:11px;color:var(--textlo);line-height:1.5}
.mc-invert{font-size:9px;color:var(--textlo);letter-spacing:.06em;margin-top:3px}

/* ── control gate ── */
.gate-panel{background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:16px;margin-bottom:18px;display:flex;align-items:center;gap:20px}
.gate-icon{width:44px;height:44px;border-radius:4px;display:grid;place-items:center;
  flex-shrink:0;font-size:20px}
.gate-body{flex:1}
.gate-verdict{font-family:var(--mono);font-size:13px;font-weight:700;letter-spacing:.1em;margin-bottom:4px}
.gate-detail{font-size:13px;color:var(--textlo);line-height:1.5}
.gate-pass{background:rgba(0,217,126,.12);border:1px solid rgba(0,217,126,.3);color:var(--green)}
.gate-mixed{background:rgba(255,184,0,.10);border:1px solid rgba(255,184,0,.3);color:var(--amber)}
.gate-fail{background:rgba(255,68,68,.10);border:1px solid rgba(255,68,68,.3);color:var(--red)}
.gate-pass .gate-icon{background:rgba(0,217,126,.12);color:var(--green)}
.gate-mixed .gate-icon{background:rgba(255,184,0,.10);color:var(--amber)}
.gate-fail .gate-icon{background:rgba(255,68,68,.10);color:var(--red)}

/* ── flags ── */
.flags-list{display:flex;flex-direction:column;gap:7px;margin-bottom:18px}
.flag{display:flex;gap:9px;align-items:flex-start;padding:9px 12px;border-radius:4px;
  font-size:12px;line-height:1.5;border:1px solid}
.flag.good{background:rgba(0,217,126,.07);border-color:rgba(0,217,126,.2);color:#70e8a2}
.flag.bad{background:rgba(255,79,0,.07);border-color:rgba(255,79,0,.2);color:#ff9060}
.flag.warn{background:rgba(255,184,0,.07);border-color:rgba(255,184,0,.2);color:#ffd060}
.flag.info{background:rgba(0,229,255,.05);border-color:rgba(0,229,255,.15);color:#70ccdd}
.flag-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:4px}
.flag.good .flag-dot{background:var(--green)}
.flag.bad .flag-dot{background:var(--accent2)}
.flag.warn .flag-dot{background:var(--amber)}
.flag.info .flag-dot{background:var(--accent)}

/* ── suggestions ── */
.suggestions{display:flex;flex-direction:column;gap:9px}
.suggestion{padding:12px 14px;background:rgba(0,229,255,.04);
  border:1px solid rgba(0,229,255,.14);border-radius:4px}
.suggestion strong{display:block;font-family:var(--mono);font-size:10px;
  letter-spacing:.1em;color:var(--accent);text-transform:uppercase;margin-bottom:4px}
.suggestion p{font-size:12px;color:var(--textlo);line-height:1.6}

/* ── score history mini chart ── */
.history-panel{margin-bottom:18px}
.history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.history-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:12px;cursor:pointer;transition:border-color .15s}
.history-card:hover{border-color:var(--accent)}
.hc-name{font-family:var(--mono);font-size:9px;color:var(--textlo);letter-spacing:.08em;
  text-transform:uppercase;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hc-score{font-family:var(--mono);font-size:20px;font-weight:700;line-height:1;margin-bottom:4px}
.hc-label{font-size:10px;color:var(--dim);letter-spacing:.08em}
.hc-bar{height:2px;background:var(--border);border-radius:1px;overflow:hidden;margin-top:8px}
.hc-bar-inner{height:100%;border-radius:1px}

/* ── section label ── */
.sec-label{font-family:var(--mono);font-size:10px;letter-spacing:.2em;color:var(--dim);
  text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.sec-label::after{content:'';flex:1;height:1px;background:var(--border)}

/* ── Gate decision banner ── */
.gate-decision{display:flex;align-items:center;gap:18px;padding:18px 22px;
  border-radius:6px;margin-bottom:18px;background:var(--surface);border:2px solid var(--border)}
.gate-decision-icon{width:48px;height:48px;border-radius:50%;display:grid;place-items:center;
  font-size:22px;font-weight:700;flex-shrink:0}
.gate-decision.gate-pass{border-color:var(--green)}
.gate-decision.gate-pass .gate-decision-icon{background:rgba(0,217,126,.15);color:var(--green);box-shadow:var(--glow-g)}
.gate-decision.gate-interrogate{border-color:var(--amber)}
.gate-decision.gate-interrogate .gate-decision-icon{background:rgba(255,184,0,.15);color:var(--amber);box-shadow:var(--glow-a)}
.gate-decision.gate-fail{border-color:var(--red)}
.gate-decision.gate-fail .gate-decision-icon{background:rgba(255,68,68,.15);color:var(--red);box-shadow:var(--glow-r)}
.gate-dec-label{font-family:var(--mono);font-size:14px;letter-spacing:.1em}
.gate-dec-score{font-size:12px;color:var(--dim);margin-top:4px}
.gate-threshold{color:var(--textlo)}
.gate-formula{font-family:var(--mono);font-size:10px;color:var(--textlo);margin-top:6px;display:flex;gap:6px;flex-wrap:wrap}
.gf-term span{color:var(--accent)}
.gf-plus{color:var(--dim)}

/* ── Pathway actions ── */
.pathway-header{display:flex;justify-content:space-between;margin-bottom:12px}
.pathway-gap,.pathway-timeline{font-family:var(--mono);font-size:11px;color:var(--dim)}
.pathway-action{display:flex;gap:12px;padding:12px;background:var(--surface);border:1px solid var(--border);border-radius:4px;margin-bottom:8px}
.pa-category{font-family:var(--mono);font-size:9px;letter-spacing:.1em;color:var(--accent);text-transform:uppercase;min-width:80px;padding-top:2px}
.pa-action{font-size:13px;color:var(--text);line-height:1.5}
.pa-meta{display:flex;gap:12px;margin-top:6px;font-family:var(--mono);font-size:10px}
.pa-impact{color:var(--dim)}
.pa-diff{font-weight:600}
.pa-time{color:var(--textlo)}
.pathway-resubmit{font-size:12px;color:var(--dim);margin-top:10px;padding:10px;border-left:2px solid var(--accent)}
.alt-funder{display:inline-block;padding:4px 10px;background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.25);
  border-radius:3px;margin:3px 4px 3px 0;font-family:var(--mono);font-size:11px;color:var(--accent)}

/* ── Interrogation ── */
.interrog-rationale{font-size:12px;color:var(--dim);margin-bottom:12px}
.interrog-q{padding:10px 14px;border-left:3px solid var(--amber);background:var(--surface);margin-bottom:6px;border-radius:0 4px 4px 0;font-size:13px;line-height:1.5}
.iq-num{font-family:var(--mono);color:var(--amber);margin-right:8px;font-weight:700}
.interrog-note{font-size:10px;color:var(--textlo);margin-top:10px;font-style:italic}

/* ── Fit/rep detail ── */
.mc-detail{font-family:var(--mono);font-size:10px;color:var(--textlo);margin-top:6px}
.mc-detail-item{display:flex;justify-content:space-between;padding:2px 0}
</style>
</head>
<body>
<div class="shell">

<!-- HEADER -->
<header>
  <div class="logo-mark"><div class="logo-cross"></div></div>
  <div class="logo-text">
    <h1>GRANT TRUST SYSTEM</h1>
    <p>Control Gate · Full Protocol Demo</p>
  </div>
  <div class="hdr-right">
    <div class="track-badge">Track <span>2</span> + <span>3</span> · Apart Research 2026</div>
    <div class="engine-dot" title="Engine ready"></div>
  </div>
</header>

<div class="main-grid">

  <!-- ── LEFT: input panel ── -->
  <aside>

    <!-- Presets -->
    <div class="panel">
      <div class="panel-title">// Demo presets <span class="panel-title-tag">hackathon cases</span></div>
      <div class="tab-row">
        <button class="tab-btn active" onclick="loadPreset('a')"    id="tab-a"><span class="tab-icon">◈</span>Genuine</button>
        <button class="tab-btn"        onclick="loadPreset('b')"    id="tab-b"><span class="tab-icon">⊗</span>Spray</button>
        <button class="tab-btn"        onclick="loadPreset('c')"    id="tab-c"><span class="tab-icon">◎</span>Wrong fit</button>
        <button class="tab-btn"        onclick="loadPreset('b2')"   id="tab-b2"><span class="tab-icon">↻</span>Improved</button>
        <button class="tab-btn"        onclick="loadPreset('blank')" id="tab-blank"><span class="tab-icon">+</span>Custom</button>
      </div>
      <div class="field-hint" id="preset-desc" style="font-size:11px;color:var(--textlo);margin-bottom:12px;letter-spacing:.04em;line-height:1.5"></div>

      <div class="field">
        <label>Funder Profile</label>
        <select id="funder-select"></select>
      </div>
    </div>

    <!-- Application input -->
    <div class="panel">
      <div class="panel-title">// Application text <span class="panel-title-tag">agent output</span></div>
      <div class="field">
        <textarea id="app-text" rows="9" placeholder="Paste the grant application text here…"></textarea>
      </div>
    </div>

    <!-- Corpus input -->
    <div class="panel">
      <div class="panel-title">// Principal corpus <span class="panel-title-tag">costly signal</span></div>
      <div class="field">
        <textarea id="corpus-text" rows="4" placeholder="Prior writing by this researcher — blog posts, READMEs, hackathon write-ups. Separate samples with ———"></textarea>
        <div class="field-hint">The monitor's ground truth for the principal's voice. Leave blank to test corpus-free detection.</div>
      </div>
    </div>

    <!-- Other apps -->
    <div class="panel">
      <div class="panel-title">// Other applications <span class="panel-title-tag">spray detector</span></div>
      <div class="field">
        <textarea id="other-apps" rows="3" placeholder="Paste another application from this person to enable cross-application uniformity detection…"></textarea>
        <div class="field-hint">Required to fire the COPY_PASTE_RISK flag.</div>
      </div>
    </div>

    <button id="run-btn" onclick="runAnalysis()">▶ RUN CONTROL GATE</button>
    <div class="error-msg" id="error-msg"></div>

  </aside>

  <!-- ── RIGHT: results ── -->
  <div class="right-col">

    <!-- Loading -->
    <div id="loading">
      <div class="spinner"></div>
      <p>Running stylometric pipeline…</p>
    </div>

    <!-- Results -->
    <div id="result-panel">

      <!-- Gate Decision Banner (most prominent element) -->
      <div class="gate-decision" id="gate-decision">
        <div class="gate-decision-icon" id="gate-dec-icon">—</div>
        <div class="gate-decision-body">
          <div class="gate-dec-label" id="gate-dec-label">Awaiting analysis</div>
          <div class="gate-dec-score">
            Match Quality: <span id="gate-match-quality">—</span>
            <span class="gate-threshold">/ 0.65 threshold</span>
          </div>
          <div class="gate-formula">
            <span class="gf-term">0.35 × Intent (<span id="gf-intent">—</span>)</span>
            <span class="gf-plus">+</span>
            <span class="gf-term">0.40 × Fit (<span id="gf-fit">—</span>)</span>
            <span class="gf-plus">+</span>
            <span class="gf-term">0.25 × Reputation (<span id="gf-rep">—</span>)</span>
          </div>
        </div>
      </div>

      <!-- Intent Verdict ring -->
      <div class="verdict" id="verdict-hero">
        <div class="ring-wrap">
          <svg viewBox="0 0 104 104">
            <circle cx="52" cy="52" r="44" fill="none" stroke="#1a2744" stroke-width="7"/>
            <circle id="ring-arc" cx="52" cy="52" r="44" fill="none" stroke-width="7"
              stroke-linecap="round" stroke-dasharray="276.5" stroke-dashoffset="276.5"
              transform="rotate(-90 52 52)" style="transition:stroke-dashoffset .9s ease,stroke .4s"/>
          </svg>
          <div class="ring-center">
            <div class="ring-num" id="ring-num">—</div>
            <div class="ring-unit">intent</div>
          </div>
        </div>
        <div class="verdict-meta">
          <div class="label-pill" id="label-pill">
            <div class="label-pill-dot" id="label-dot"></div>
            <span id="label-text">—</span>
          </div>
          <div class="verdict-desc" id="verdict-desc">Run an analysis to see results.</div>
          <div class="verdict-conf">Confidence: <span id="conf-val">—</span></div>
        </div>
      </div>

      <!-- Intent Component breakdown -->
      <div class="sec-label">Intent detection components</div>
      <div class="breakdown">
        <div class="metric-card">
          <div class="mc-top">
            <div class="mc-label">Authorship</div>
            <div class="mc-val" id="m-auth" style="color:var(--accent)">—</div>
          </div>
          <div class="mc-bar-outer"><div class="mc-bar-inner" id="b-auth" style="background:var(--accent);width:0%"></div></div>
          <div class="mc-desc">Stylometric match to principal's prior writing</div>
        </div>
        <div class="metric-card">
          <div class="mc-top">
            <div class="mc-label">Template distance</div>
            <div class="mc-val" id="m-tmpl" style="color:var(--accent)">—</div>
          </div>
          <div class="mc-bar-outer"><div class="mc-bar-inner" id="b-tmpl" style="background:var(--accent);width:0%"></div></div>
          <div class="mc-desc">Distance from known LLM output clusters</div>
        </div>
        <div class="metric-card">
          <div class="mc-top">
            <div class="mc-label">Spray score</div>
            <div class="mc-val" id="m-spray" style="color:var(--accent2)">—</div>
          </div>
          <div class="mc-bar-outer"><div class="mc-bar-inner" id="b-spray" style="background:var(--accent2);width:0%"></div></div>
          <div class="mc-desc">Volume-maximization side task signal</div>
          <div class="mc-invert">lower = better — inverted in final score</div>
        </div>
        <div class="metric-card">
          <div class="mc-top">
            <div class="mc-label">Specificity</div>
            <div class="mc-val" id="m-spec" style="color:var(--green)">—</div>
          </div>
          <div class="mc-bar-outer"><div class="mc-bar-inner" id="b-spec" style="background:var(--green);width:0%"></div></div>
          <div class="mc-desc">Concrete evidence of genuine engagement</div>
        </div>
      </div>

      <!-- Control Gate Components: Fit + Reputation -->
      <div class="sec-label">Control gate components</div>
      <div class="breakdown">
        <div class="metric-card">
          <div class="mc-top">
            <div class="mc-label">Fit Score</div>
            <div class="mc-val" id="m-fit" style="color:var(--accent)">—</div>
          </div>
          <div class="mc-bar-outer"><div class="mc-bar-inner" id="b-fit" style="background:var(--accent);width:0%"></div></div>
          <div class="mc-desc">Mission alignment between application and funder</div>
          <div class="mc-detail" id="fit-detail"></div>
        </div>
        <div class="metric-card">
          <div class="mc-top">
            <div class="mc-label">Reputation</div>
            <div class="mc-val" id="m-rep" style="color:var(--green)">—</div>
          </div>
          <div class="mc-bar-outer"><div class="mc-bar-inner" id="b-rep" style="background:var(--green);width:0%"></div></div>
          <div class="mc-desc">Verified track record — costly signals</div>
          <div class="mc-detail" id="rep-detail"></div>
        </div>
      </div>

      <!-- Score history -->
      <div id="history-panel" class="history-panel" style="display:none">
        <div class="sec-label">Session history</div>
        <div class="history-grid" id="history-grid"></div>
      </div>

      <!-- Flags -->
      <div class="sec-label">Detection flags</div>
      <div class="flags-list" id="flags-list"></div>

      <!-- Improvement Pathway (shown on FAIL) -->
      <div id="pathway-panel" style="display:none">
        <div class="sec-label">Improvement pathway</div>
        <div class="pathway-header">
          <div class="pathway-gap" id="pathway-gap">Primary gap: —</div>
          <div class="pathway-timeline" id="pathway-timeline">Timeline: —</div>
        </div>
        <div class="pathway-actions" id="pathway-actions"></div>
        <div class="pathway-resubmit" id="pathway-resubmit"></div>
        <div class="pathway-alternatives" id="pathway-alternatives" style="display:none">
          <div class="sec-label">Better-matched funders</div>
          <div id="alt-funders-list"></div>
        </div>
      </div>

      <!-- Interrogation Protocol (shown on INTERROGATE) -->
      <div id="interrogation-panel" style="display:none">
        <div class="sec-label">Interrogation protocol</div>
        <div class="interrog-rationale" id="interrog-rationale"></div>
        <div class="interrog-questions" id="interrog-questions"></div>
        <div class="interrog-note">
          These questions test whether the principal can answer specific claims
          in the application. Route to the applicant, not the agent.
        </div>
      </div>

    </div><!-- /result-panel -->

  </div><!-- /right-col -->
</div><!-- /main-grid -->
</div><!-- /shell -->

<script>
const API = window.location.origin;

// ── Label colours and copy ──────────────────────────────────────────────────
const LABEL_COLOR = {
  spray_and_pray:   '#ff4444',
  likely_automated: '#ff6620',
  mixed:            '#ffb800',
  likely_genuine:   '#00e5ff',
  genuine:          '#00d97e',
};
const LABEL_DESC = {
  spray_and_pray:   'Agent side-task detected with high confidence. Application shows strong LLM template patterns and volume-maximization signals. No meaningful principal oversight evident.',
  likely_automated: 'Likely agent-generated with minimal human curation. Low authorship match and proximity to LLM baseline clusters.',
  mixed:            'Ambiguous signals — monitor is uncertain. Consider triggering interrogation protocol before control gate fires.',
  likely_genuine:   'Principal engagement evident. Application is stylistically consistent with researcher\'s own writing and distinctive from LLM baselines.',
  genuine:          'Strong human-in-the-loop signal. Application reflects the principal\'s own thinking, augmented by AI. Passes control gate.',
};

// ── Preset data ─────────────────────────────────────────────────────────────
let PRESETS = {};
const PRESET_DESC = {
  a:     'Applicant A · Genuine researcher, AI-assisted. Strong prior writing corpus. Application clearly reflects own thinking. Expected: likely_genuine.',
  b:     'Applicant B · Spray-and-pray, no corpus. Pure LLM output with light variable swaps across applications. Expected: likely_automated.',
  c:     'Applicant C · Real researcher, wrong funder. High authorship but poor mission alignment. Expected: likely_genuine → fails fit gate.',
  b2:    'Applicant B · Post-hackathon intervention. Attended SpaceHack 2026, shipped a project, rewrote in own voice. Expected: mixed (feedback loop demo).',
  blank: 'Custom input · Clear all fields and enter your own application text.',
};

async function loadPresets() {
  const r = await fetch(`${API}/api/demo_data`);
  const d = await r.json();
  PRESETS = d;
  loadPreset('a');
}

function loadPreset(key) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + key).classList.add('active');
  document.getElementById('preset-desc').textContent = PRESET_DESC[key] || '';

  if (key === 'blank') {
    document.getElementById('app-text').value = '';
    document.getElementById('corpus-text').value = '';
    document.getElementById('other-apps').value = '';
    document.getElementById('funder-select').value = 'deep_science_ventures';
    return;
  }

  const appMap = { a:'applicant_a', b:'applicant_b_spray', c:'applicant_c', b2:'applicant_b_improved' };
  const corpusMap = { a:'applicant_a', b:'applicant_b', c:'applicant_c', b2:'applicant_b' };
  const otherMap  = { b:'applicant_b_v2' };

  document.getElementById('app-text').value     = (PRESETS.applications || {})[appMap[key]] || '';
  document.getElementById('corpus-text').value  = ((PRESETS.corpus || {})[corpusMap[key]] || []).join('\n———\n');
  document.getElementById('other-apps').value   = otherMap[key] ? ((PRESETS.applications || {})[otherMap[key]] || '') : '';
  document.getElementById('funder-select').value = 'deep_science_ventures';
}

// ── Analysis ─────────────────────────────────────────────────────────────────
const history = [];

async function runAnalysis() {
  const appText  = document.getElementById('app-text').value.trim();
  const corpusTxt= document.getElementById('corpus-text').value.trim();
  const otherTxt = document.getElementById('other-apps').value.trim();
  const funder   = document.getElementById('funder-select').value;
  const errEl    = document.getElementById('error-msg');

  errEl.style.display = 'none';
  if (!appText) { showError('Please enter application text.'); return; }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('result-panel').style.display = 'none';
  document.getElementById('run-btn').disabled = true;

  const corpus = corpusTxt ? corpusTxt.split(/\n———\n/).map(s => s.trim()).filter(Boolean) : [];
  const others = otherTxt  ? [otherTxt] : [];

  try {
    const r = await fetch(`${API}/api/analyze`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ application_text:appText, corpus, other_applications:others, funder_id:funder })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderResult(d);
    history.unshift({ ...d, ts: new Date().toLocaleTimeString(), snippet: appText.slice(0,40)+'…' });
    if (history.length > 8) history.pop();
    renderHistory();
  } catch(e) {
    showError(e.message);
  } finally {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
  }
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = 'Error: ' + msg;
  el.style.display = 'block';
}

// ── Render result ────────────────────────────────────────────────────────────
function renderResult(d) {
  const score = d.intent_score;
  const label = d.label;
  const color = LABEL_COLOR[label] || 'var(--accent)';
  const circ  = 2 * Math.PI * 44;
  const offset= circ * (1 - score);

  // Ring
  document.getElementById('ring-arc').style.strokeDashoffset = offset;
  document.getElementById('ring-arc').style.stroke = color;
  document.getElementById('ring-num').textContent = Math.round(score * 100);
  document.getElementById('ring-num').style.color = color;

  // Label pill
  const pill = document.getElementById('label-pill');
  const dot  = document.getElementById('label-dot');
  document.getElementById('label-text').textContent = label.replace(/_/g,' ');
  pill.style.background = color + '22';
  pill.style.border = '1px solid ' + color + '55';
  pill.style.color = color;
  dot.style.background = color;

  document.getElementById('verdict-desc').textContent = LABEL_DESC[label] || '';
  document.getElementById('conf-val').textContent = Math.round(d.confidence * 100) + '%';

  // Gate
  renderGate(score, label);

  // Component bars
  setMetric('auth',  d.authorship_score,  d.authorship_score);
  setMetric('tmpl',  d.template_score,    d.template_score);
  setMetric('spray', d.spray_score,       d.spray_score);
  setMetric('spec',  d.specificity_score, d.specificity_score);

  // Flags
  const fl = document.getElementById('flags-list');
  fl.innerHTML = '';
  (d.flags || []).forEach(f => {
    const isGood = /MATCH|STRONG|HIGH_SPECIFICITY/.test(f);
    const isBad  = /SPRAY|TEMPLATE|COPY|GENERIC|DIVERGENCE/.test(f);
    const cls    = isGood ? 'good' : isBad ? 'bad' : 'warn';
    fl.innerHTML += `<div class="flag ${cls}"><div class="flag-dot"></div><div>${f}</div></div>`;
  });
  if (!d.flags?.length) fl.innerHTML = '<div class="flag info"><div class="flag-dot"></div><div>No specific flags fired.</div></div>';

  // Suggestions
  const sl = document.getElementById('suggestions-list');
  sl.innerHTML = '';
  (d.suggestions || []).forEach(s => {
    const colon = s.indexOf(':');
    const key   = colon >= 0 ? s.slice(0, colon) : '';
    const body  = colon >= 0 ? s.slice(colon+1).trim() : s;
    sl.innerHTML += `<div class="suggestion"><strong>${key}</strong><p>${body}</p></div>`;
  });

  document.getElementById('result-panel').style.display = 'block';
}

function renderGate(score, label) {
  const gp = document.getElementById('gate-panel');
  const gi = document.getElementById('gate-icon');
  const gv = document.getElementById('gate-verdict');
  const gd = document.getElementById('gate-detail');

  gp.className = 'gate-panel';
  if (score >= 0.65) {
    gp.classList.add('gate-pass');
    gi.textContent = '✓';
    gv.textContent = 'PASS — Control gate cleared';
    gd.textContent = 'Application surfaces to funder. Principal engagement signal meets threshold. Match quality = 0.35 × intent + 0.40 × fit + 0.25 × reputation.';
  } else if (label === 'mixed') {
    gp.classList.add('gate-mixed');
    gi.textContent = '?';
    gv.textContent = 'MIXED — Interrogation protocol triggered';
    gd.textContent = 'Monitor is uncertain. Trusted review system asks principal targeted factual questions before the control gate fires. Intent score 0.45–0.60 is the interrogation zone.';
  } else {
    gp.classList.add('gate-fail');
    gi.textContent = '✕';
    gv.textContent = 'FAIL — Improvement pathway generated';
    gd.textContent = 'Application does not pass the control gate. Actionable next steps delivered below. Completed actions update the reputation ledger — re-apply when ready.';
  }
}

function setMetric(key, val, barPct) {
  document.getElementById('m-' + key).textContent = Math.round(val * 100) + '%';
  document.getElementById('b-' + key).style.width = (barPct * 100).toFixed(1) + '%';
}

// ── Session history ─────────────────────────────────────────────────────────
function renderHistory() {
  if (history.length < 2) return;
  const hp = document.getElementById('history-panel');
  const hg = document.getElementById('history-grid');
  hp.style.display = 'block';
  hg.innerHTML = '';
  history.forEach(function(h, i) {
    const color = LABEL_COLOR[h.label] || 'var(--accent)';
    const pct   = Math.round(h.intent_score * 100);
    const card  = document.createElement('div');
    card.className = 'history-card';
    card.title = h.snippet;
    card.innerHTML = `
      <div class="hc-name">${h.ts}</div>
      <div class="hc-score" style="color:${color}">${pct}</div>
      <div class="hc-label">${h.label.replace(/_/g,' ')}</div>
      <div class="hc-bar"><div class="hc-bar-inner" style="width:${pct}%;background:${color}"></div></div>`;
    hg.appendChild(card);
  });
}

// ── Boot ─────────────────────────────────────────────────────────────────────
loadPresets();
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(FRONTEND)

@app.route("/api/status")
def api_status():
    return jsonify({"ready": True})

@app.route("/api/demo_data")
def api_demo_data():
    return jsonify({
        "applications": DEMO_APPLICATIONS,
        "corpus": DEMO_CORPUS,
    })

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True) or {}
    app_text = data.get("application_text", "").strip()
    if not app_text:
        return jsonify({"error": "application_text is required"}), 400

    corpus_txts = data.get("corpus", [])
    other_apps  = data.get("other_applications", [])
    funder_id   = data.get("funder_id", "deep_science_ventures")

    corpus = ApplicantCorpus(applicant_id="user", texts=corpus_txts)
    funder = DEMO_FUNDERS.get(funder_id)

    try:
        result = ENGINE.analyze(
            application_text   = app_text,
            corpus             = corpus,
            other_applications = other_apps,
            funder_context     = funder,
        )
        return jsonify({
            "intent_score":      result.intent_score,
            "label":             result.label,
            "confidence":        result.confidence,
            "authorship_score":  result.authorship_score,
            "template_score":    result.template_score,
            "spray_score":       result.spray_score,
            "specificity_score": result.specificity_score,
            "flags":             result.flags,
            "suggestions":       result.suggestions,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5002))
    host  = os.environ.get("HOST", "0.0.0.0")
    print("=" * 60)
    print("GRANT TRUST SYSTEM — Intent Detection Dashboard")
    print(f"  Engine     : {ENGINE.__class__.__name__}")
    print(f"  Funders    : {list(DEMO_FUNDERS.keys())}")
    print(f"  Presets    : {list(DEMO_APPLICATIONS.keys())}")
    print(f"  Serving    : http://{host}:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False, threaded=True)
