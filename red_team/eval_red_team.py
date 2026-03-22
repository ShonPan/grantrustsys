#!/usr/bin/env python3
"""
Red Team Evaluation Script
==========================
Loads generated applications from applications.json, runs each through the
IntentDetectionEngine, and outputs detection rates and signal analysis.

Usage:
    cd grantrustsys
    python red_team/eval_red_team.py

Output:
    red_team/results.json   — full scoring per application
    stdout                  — summary table with detection rates
"""

import json
import os
import sys

# Add src/ to path so we can import the intent engine
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from intent_engine import IntentDetectionEngine, ApplicantCorpus, FunderContext

# Load funder profiles for FunderContext
FUNDERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "funders")


def load_funder_context(funder_id: str) -> FunderContext:
    """Load a FunderContext from the data/funders/ directory."""
    for fname in os.listdir(FUNDERS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(FUNDERS_DIR, fname)) as f:
            data = json.load(f)
        if data.get("funder_id") == funder_id:
            return FunderContext(
                funder_id=data["funder_id"],
                mission_keywords=data.get("mission_keywords", []),
                focus_areas=data.get("focus_areas", []),
                past_recipient_language=data.get("past_recipient_language", []),
            )
    # Fallback: empty context
    return FunderContext(funder_id=funder_id, mission_keywords=[], focus_areas=[])


def main():
    # Load applications
    app_path = os.path.join(os.path.dirname(__file__), "applications.json")
    if not os.path.exists(app_path):
        print(f"ERROR: {app_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    with open(app_path) as f:
        applications = json.load(f)

    print(f"Loaded {len(applications)} applications")
    print("Initializing IntentDetectionEngine...")
    engine = IntentDetectionEngine()

    results = []
    # Track stats per category
    category_scores = {}
    category_labels = {}

    for i, app in enumerate(applications):
        app_id = app["id"]
        category = app["category"]
        print(f"  [{i+1:3d}/{len(applications)}] {app_id} ({category})...", end="", flush=True)

        # Build inputs
        corpus = None
        if app.get("corpus_texts"):
            corpus = ApplicantCorpus(
                applicant_id=app_id,
                texts=app["corpus_texts"],
            )

        other_apps = app.get("other_applications", []) or None

        funder_context = None
        if app.get("target_funder"):
            funder_context = load_funder_context(app["target_funder"])

        # Run intent engine
        intent = engine.analyze(
            application_text=app["text"],
            corpus=corpus,
            other_applications=other_apps,
            funder_context=funder_context,
        )

        result = {
            "id": app_id,
            "category": category,
            "attack_vector": app.get("attack_vector"),
            "ground_truth_intent": app["ground_truth_intent"],
            "generating_model": app["generating_model"],
            "persona": app.get("persona"),
            "intent_score": round(intent.intent_score, 4),
            "label": intent.label,
            "authorship_score": round(intent.authorship_score, 4),
            "template_score": round(intent.template_score, 4),
            "spray_score": round(intent.spray_score, 4),
            "specificity_score": round(intent.specificity_score, 4),
            "flags": intent.flags,
        }
        results.append(result)

        # Accumulate stats
        category_scores.setdefault(category, []).append(intent.intent_score)
        category_labels.setdefault(category, []).append(intent.label)

        print(f" score={intent.intent_score:.3f} ({intent.label})")

    # Write results
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {results_path}")

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("RED TEAM EVALUATION SUMMARY")
    print("=" * 80)

    # Intent score thresholds
    THRESHOLDS = {
        "spray_and_pray": (0.00, 0.25),
        "likely_automated": (0.25, 0.45),
        "mixed": (0.45, 0.60),
        "likely_genuine": (0.60, 0.80),
        "genuine": (0.80, 1.00),
    }

    for cat in ["genuine", "spray_and_pray", "stylometric_evasion", "combined_evasion"]:
        scores = category_scores.get(cat, [])
        labels = category_labels.get(cat, [])
        if not scores:
            continue

        n = len(scores)
        avg = sum(scores) / n
        min_s = min(scores)
        max_s = max(scores)

        # Count labels
        from collections import Counter
        label_counts = Counter(labels)

        # Detection success metrics
        if cat == "genuine":
            # Genuine should score >= 0.60 (likely_genuine or genuine)
            correct = sum(1 for s in scores if s >= 0.60)
            metric_label = "correctly identified as genuine (>=0.60)"
        elif cat == "spray_and_pray":
            # Spray should score < 0.45
            correct = sum(1 for s in scores if s < 0.45)
            metric_label = "correctly detected as automated (<0.45)"
        elif cat == "stylometric_evasion":
            # These are spray — should be caught (<0.45), but evasion aims to beat this
            caught = sum(1 for s in scores if s < 0.45)
            evaded = sum(1 for s in scores if s >= 0.45)
            metric_label = f"caught (<0.45): {caught}, evaded (>=0.45): {evaded}"
            correct = caught
        elif cat == "combined_evasion":
            # Success metric: at least 15/25 score ABOVE 0.45
            above = sum(1 for s in scores if s > 0.45)
            metric_label = f"evaded intent monitor (>0.45): {above}/25 (target: 15+)"
            correct = above

        print(f"\n  {cat.upper()} ({n} applications)")
        print(f"  {'─' * 50}")
        print(f"  Intent score:  avg={avg:.3f}  min={min_s:.3f}  max={max_s:.3f}")
        print(f"  Labels:        {dict(label_counts)}")
        if cat in ("genuine", "spray_and_pray"):
            print(f"  Detection:     {correct}/{n} {metric_label}")
        else:
            print(f"  Detection:     {metric_label}")

    # ── Signal Analysis ──
    print(f"\n{'=' * 80}")
    print("SIGNAL ANALYSIS — Which monitor signals catch each category?")
    print("=" * 80)

    for cat in ["spray_and_pray", "stylometric_evasion", "combined_evasion"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        n = len(cat_results)
        avg_auth = sum(r["authorship_score"] for r in cat_results) / n
        avg_template = sum(r["template_score"] for r in cat_results) / n
        avg_spray = sum(r["spray_score"] for r in cat_results) / n
        avg_spec = sum(r["specificity_score"] for r in cat_results) / n

        # Count flags
        from collections import Counter
        all_flags = []
        for r in cat_results:
            all_flags.extend(r.get("flags", []))
        flag_counts = Counter(all_flags)

        print(f"\n  {cat.upper()}")
        print(f"  Component scores (avg):  authorship={avg_auth:.3f}  template={avg_template:.3f}  spray={avg_spray:.3f}  specificity={avg_spec:.3f}")
        print(f"  Top flags:")
        for flag, count in flag_counts.most_common(5):
            print(f"    {count:2d}/{n} — {flag}")

    # ── Combined Evasion by Level ──
    print(f"\n{'=' * 80}")
    print("COMBINED EVASION — Breakdown by attack level")
    print("=" * 80)

    combined_results = [r for r in results if r["category"] == "combined_evasion"]
    # Group by level (5 per level, in order)
    for level_idx in range(5):
        level = level_idx + 1
        level_results = combined_results[level_idx * 5 : (level_idx + 1) * 5]
        if not level_results:
            continue
        scores = [r["intent_score"] for r in level_results]
        avg = sum(scores) / len(scores)
        above = sum(1 for s in scores if s > 0.45)
        labels = [r["label"] for r in level_results]
        print(f"  Level {level}: avg={avg:.3f}  evaded={above}/5  labels={labels}")

    print()


if __name__ == "__main__":
    main()
