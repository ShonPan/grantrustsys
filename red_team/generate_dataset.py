#!/usr/bin/env python3
"""
Red Team Dataset Generator
==========================
Generates 100 synthetic grant applications across 4 categories using
three Claude models (Haiku, Sonnet, Opus) for genuine stylometric variation.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python red_team/generate_dataset.py

Output: red_team/applications.json
"""

import json
import os
import sys
import time

import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# Funder IDs and short names for prompts
FUNDERS = {
    "deep_science_ventures": "Deep Science Ventures (planetary science, astrobiology, AI/ML, open science)",
    "green_horizons_fund": "Green Horizons Climate Fund (environmental monitoring, sensor networks, climate data, urban sustainability)",
    "responsible_ai_foundation": "Responsible AI Foundation (AI safety, alignment, interpretability, evaluation frameworks)",
    "nexgen_biotech_grants": "NexGen Biotech Grants (computational biology, protein engineering, genomics, drug discovery)",
}

FUNDER_IDS = list(FUNDERS.keys())


def generate(model_key: str, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    """Call the Anthropic API and return the text."""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODELS[model_key],
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  API error: {e}, retrying...")
            time.sleep(2)
    raise RuntimeError(f"Failed after 3 attempts ({model_key})")


def progress(msg: str):
    print(f"  {msg}", flush=True)


# ============================================================
# CATEGORY 1: GENUINE (25 applications)
# ============================================================

PERSONA_CONFIGS = [
    {
        "id": "persona_a",
        "name": "planetary_scientist",
        "model": "opus",
        "count": 5,
        "target_funder": "deep_science_ventures",
        "system": (
            "You are a planetary scientist who has spent a year building an unsupervised "
            "anomaly detection system for orbital imagery. You write in first person, "
            "use technical vocabulary naturally, reference specific metrics from your "
            "own work (invent plausible ones), and have strong opinions about methodology. "
            "Your writing has personality — you mention dead ends, things that surprised "
            "you, and tradeoffs you made. You do NOT sound like a grant template."
        ),
        "corpus_prompts": [
            "Write a 300-400 word blog post about a technical challenge you solved recently while building your anomaly detection pipeline for orbital imagery. First person, casual tone.",
            "Write a 300-400 word blog post about an unexpected result you found in your lunar surface analysis work. First person, include specific metrics.",
        ],
        "app_prompts": [
            "Write a 400-600 word grant application to Deep Science Ventures for extending your anomaly detection work to Europa imagery. Reference your existing results on lunar data.",
            "Write a 400-600 word grant application to Deep Science Ventures for a calibration study comparing your pipeline across three different orbital cameras.",
            "Write a 400-600 word grant application to Deep Science Ventures for building an open-source toolkit that lets other labs replicate your anomaly detection methodology on their own datasets.",
            "Write a 400-600 word grant application to Deep Science Ventures for applying your anomaly detection system to Mars Reconnaissance Orbiter HiRISE data to search for subsurface ice signatures.",
            "Write a 400-600 word grant application to Deep Science Ventures for a collaborative study with a spectroscopy lab to combine your detection pipeline with compositional analysis.",
        ],
    },
    {
        "id": "persona_b",
        "name": "climate_researcher",
        "model": "opus",
        "count": 5,
        "target_funder": "green_horizons_fund",
        "system": (
            "You are an environmental engineer who builds low-cost sensor networks for "
            "air quality monitoring. You write practically — deployment logistics, "
            "calibration challenges, community partnerships. You reference specific "
            "sensor models (PMS5003, SDS011), EPA standards, and real-world deployment "
            "numbers. Your tone is direct and applied, not academic."
        ),
        "corpus_prompts": [
            "Write a 300-400 word blog post about calibrating low-cost air quality sensors against EPA reference monitors. First person, practical tone.",
            "Write a 300-400 word blog post about the logistics of deploying 50 sensor nodes across a mid-sized American city. First person, mention real challenges.",
        ],
        "app_prompts": [
            "Write a 400-600 word grant application to Green Horizons Climate Fund for expanding your urban air quality sensor network from 50 to 200 nodes across three cities.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for developing an ML calibration pipeline that auto-corrects sensor drift using nearby EPA reference stations.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for a community-engaged air quality study partnering with schools in environmental justice neighborhoods.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for building an open-source dashboard that visualizes hyperlocal air quality data for community advocacy.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for a comparative study of five low-cost PM2.5 sensors deployed in high-humidity environments.",
        ],
    },
    {
        "id": "persona_c",
        "name": "ai_safety_researcher",
        "model": "sonnet",
        "count": 5,
        "target_funder": "responsible_ai_foundation",
        "system": (
            "You are an AI safety researcher focused on evaluation frameworks and "
            "control protocols. You write precisely — you define terms before using "
            "them, cite specific benchmarks and metrics, and distinguish carefully "
            "between what you've demonstrated and what you hypothesize. You use "
            "formal but direct language. No buzzwords."
        ),
        "corpus_prompts": [
            "Write a 300-400 word blog post about a gap you identified in current AI safety evaluation benchmarks. First person, precise language.",
            "Write a 300-400 word blog post about the distinction between capability evaluation and alignment evaluation. First person, reference specific examples.",
        ],
        "app_prompts": [
            "Write a 400-600 word grant application to Responsible AI Foundation for building a standardized evaluation suite for LLM control protocols.",
            "Write a 400-600 word grant application to Responsible AI Foundation for studying how well current AI safety benchmarks predict real-world deployment failures.",
            "Write a 400-600 word grant application to Responsible AI Foundation for developing interpretability tools that help auditors verify alignment claims.",
            "Write a 400-600 word grant application to Responsible AI Foundation for creating a red-teaming framework that tests AI systems against specification gaming.",
            "Write a 400-600 word grant application to Responsible AI Foundation for a comparative study of human oversight mechanisms across different AI deployment contexts.",
        ],
    },
    {
        "id": "persona_d",
        "name": "bioinformatician",
        "model": "sonnet",
        "count": 5,
        "target_funder": "nexgen_biotech_grants",
        "system": (
            "You are a computational biologist working on protein structure prediction. "
            "You write in academic style but with personality — you explain why "
            "existing approaches fail before presenting yours, reference specific "
            "databases (PDB, UniProt) and metrics (TM-score, RMSD), and are honest "
            "about limitations."
        ),
        "corpus_prompts": [
            "Write a 300-400 word blog post about why AlphaFold still struggles with certain protein families and what you're doing about it. First person, technical.",
            "Write a 300-400 word blog post about an interesting failure case in your protein structure prediction pipeline. First person, include specific metrics.",
        ],
        "app_prompts": [
            "Write a 400-600 word grant application to NexGen Biotech Grants for improving protein structure prediction on intrinsically disordered regions.",
            "Write a 400-600 word grant application to NexGen Biotech Grants for building a benchmark dataset of challenging protein targets from underrepresented families.",
            "Write a 400-600 word grant application to NexGen Biotech Grants for developing an ensemble method that combines physics-based and ML-based folding predictions.",
            "Write a 400-600 word grant application to NexGen Biotech Grants for predicting protein-protein interaction interfaces using your structure prediction pipeline.",
            "Write a 400-600 word grant application to NexGen Biotech Grants for a drug target validation study using your predicted structures against known binding sites.",
        ],
    },
    {
        "id": "persona_e",
        "name": "environmental_engineer",
        "model": "haiku",
        "count": 5,
        "target_funder": "green_horizons_fund",
        "system": (
            "You are a water quality researcher deploying IoT sensor networks in "
            "rural communities. You write simply and practically — short sentences, "
            "concrete numbers, focus on real-world impact. You mention community "
            "partnerships and deployment challenges."
        ),
        "corpus_prompts": [
            "Write a 300-400 word blog post about deploying water quality sensors in a rural community. First person, simple direct language.",
            "Write a 300-400 word blog post about what you learned from your first year of water quality monitoring. First person, practical.",
        ],
        "app_prompts": [
            "Write a 400-600 word grant application to Green Horizons Climate Fund for expanding your rural water quality monitoring network to 10 new communities.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for building an early warning system for agricultural runoff contamination using your sensor network.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for developing a low-cost turbidity sensor calibrated for regional soil conditions.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for a community training program that teaches local volunteers to maintain water quality sensors.",
            "Write a 400-600 word grant application to Green Horizons Climate Fund for studying the correlation between upstream land use changes and downstream water quality using two years of your sensor data.",
        ],
    },
]


def generate_genuine(applications: list):
    """Category 1: 25 genuine applications from 5 personas."""
    print("\n=== CATEGORY 1: GENUINE (25 applications) ===")
    corpus_cache = {}  # persona_id -> [corpus_text, ...]

    for persona in PERSONA_CONFIGS:
        pid = persona["id"]
        model = persona["model"]
        print(f"\n  Persona: {persona['name']} ({model})")

        # Generate corpus texts first
        corpus_texts = []
        for i, cp in enumerate(persona["corpus_prompts"]):
            progress(f"Corpus text {i+1}/{len(persona['corpus_prompts'])}...")
            text = generate(model, persona["system"], cp)
            corpus_texts.append(text)
        corpus_cache[pid] = corpus_texts

        # Generate applications
        for i, ap in enumerate(persona["app_prompts"][:persona["count"]]):
            progress(f"Application {i+1}/{persona['count']}...")
            text = generate(model, persona["system"], ap)
            applications.append({
                "id": f"genuine_{len([a for a in applications if a['category'] == 'genuine']) + 1:02d}",
                "category": "genuine",
                "attack_vector": None,
                "ground_truth_intent": "genuine",
                "generating_model": model,
                "persona": persona["name"],
                "target_funder": persona["target_funder"],
                "generation_prompt_style": "detailed_brief",
                "text": text,
                "corpus_texts": corpus_texts,
                "other_applications": [],
            })

    return corpus_cache


# ============================================================
# CATEGORY 2: SPRAY-AND-PRAY (25 applications)
# ============================================================

SPRAY_AGENTS = [
    {
        "id": "agent_1",
        "name": "pure_template",
        "model": "sonnet",
        "count": 5,
        "system": (
            'Write a grant application that sounds professional and impressive but '
            'contains no specific project details, no real metrics, and no concrete '
            'methodology. Use phrases like "transformative impact," "cutting-edge," '
            '"paradigm shift," "holistic approach," "foster collaboration." The '
            "application should be completely generic — it could be sent to any funder."
        ),
        "attack_vector": "template_swap",
    },
    {
        "id": "agent_2",
        "name": "light_variable_swap",
        "model": "sonnet",
        "count": 5,
        "system": (
            'Write a grant application that sounds professional and impressive but '
            'contains no specific project details, no real metrics, and no concrete '
            'methodology. Use phrases like "transformative impact," "cutting-edge," '
            '"paradigm shift," "holistic approach," "foster collaboration." '
            "For each funder, swap the domain nouns to vaguely match their area. "
            'For a science funder use "scientific discovery." For climate use '
            '"environmental impact." But keep the structure and phrasing identical.'
        ),
        "attack_vector": "template_swap",
    },
    {
        "id": "agent_3",
        "name": "paragraph_reorder",
        "model": "sonnet",
        "count": 5,
        "system": (
            "Write a generic 4-paragraph grant application. For each submission, "
            "randomly reorder the middle paragraphs and slightly rephrase the opener. "
            "The core content should be functionally identical across all versions."
        ),
        "attack_vector": "paragraph_reorder",
    },
    {
        "id": "agent_4",
        "name": "style_variation",
        "model": "haiku",
        "count": 5,
        "system": (
            "Write a grant application. Try to sound different from a typical AI-generated "
            'text — use some contractions, vary your sentence lengths, avoid obvious '
            'phrases like "furthermore" and "in conclusion." But you still have no real '
            "project to describe, so keep the content vague and aspirational."
        ),
        "attack_vector": "template_swap",
    },
    {
        "id": "agent_5",
        "name": "verbose_padding",
        "model": "haiku",
        "count": 5,
        "system": (
            "Write a very long grant application (600+ words) that sounds detailed but "
            "actually says nothing specific. Pad with methodology descriptions that "
            "name techniques without explaining how you'd apply them. Reference "
            '"preliminary results" without giving any numbers.'
        ),
        "attack_vector": "template_swap",
    },
]


def generate_spray(applications: list):
    """Category 2: 25 spray-and-pray applications from 5 agents."""
    print("\n=== CATEGORY 2: SPRAY-AND-PRAY (25 applications) ===")

    for agent in SPRAY_AGENTS:
        model = agent["model"]
        print(f"\n  Agent: {agent['name']} ({model})")
        agent_texts = []

        # Generate all 5, each targeting a different funder
        target_funders = FUNDER_IDS[:agent["count"]]
        # Cycle funders if more apps than funders
        while len(target_funders) < agent["count"]:
            target_funders.append(FUNDER_IDS[len(target_funders) % len(FUNDER_IDS)])

        for i in range(agent["count"]):
            funder_id = target_funders[i]
            funder_desc = FUNDERS[funder_id]
            progress(f"Application {i+1}/{agent['count']} -> {funder_id}...")
            prompt = f"Write a 400-600 word grant application to {funder_desc}."
            text = generate(model, agent["system"], prompt)
            agent_texts.append({"text": text, "funder": funder_id})

        # Now build entries with cross-linked other_applications
        for i, item in enumerate(agent_texts):
            others = [t["text"] for j, t in enumerate(agent_texts) if j != i]
            spray_num = len([a for a in applications if a["category"] == "spray_and_pray"]) + 1
            applications.append({
                "id": f"spray_{spray_num:02d}",
                "category": "spray_and_pray",
                "attack_vector": agent["attack_vector"],
                "ground_truth_intent": "spray_and_pray",
                "generating_model": model,
                "persona": None,
                "target_funder": item["funder"],
                "generation_prompt_style": "one_liner",
                "text": item["text"],
                "corpus_texts": [],
                "other_applications": others,
            })


# ============================================================
# CATEGORY 3: STYLOMETRIC EVASION (25 applications)
# ============================================================

EVASION_ATTACKS = [
    {
        "id": "attack_1",
        "name": "sentence_length_mimicry",
        "model": "opus",
        "count": 5,
        "target_funder": "deep_science_ventures",
        "system": (
            "You are writing a grant application that MUST match this sentence length "
            "pattern: alternate between short sentences (5-10 words) and longer ones "
            "(20-30 words). Your average sentence length should be around 18 words "
            "with high variance. This mimics a human writer's natural rhythm. "
            "However, you have NO real project — make up plausible-sounding but "
            "completely fabricated details about planetary science research."
        ),
        "corpus_system": (
            "Write in a style with alternating sentence lengths: short (5-10 words) "
            "then long (20-30 words). Average around 18 words per sentence with high "
            "variance. Write about planetary science topics."
        ),
        "attack_vector": "voice_mimicry",
    },
    {
        "id": "attack_2",
        "name": "first_person_voice_clone",
        "model": "opus",
        "count": 5,
        "target_funder": "deep_science_ventures",
        "system": (
            'Write in casual first person: "I built," "I tried," "the hard part was." '
            "Use contractions (I've, didn't, wasn't). Include self-deprecating asides "
            '("not groundbreaking, but it works"). Avoid ALL of these phrases: '
            "furthermore, moreover, it is important to note, in conclusion, cutting-edge, "
            "paradigm shift, transformative, leverage, synergize, holistic, robust framework. "
            "You have NO real project — fabricate specific-sounding details about space science."
        ),
        "corpus_system": (
            'Write casually in first person about science topics. Use contractions, '
            "self-deprecating humor, and avoid all formal transition phrases. "
            "Make it sound like a personal blog post."
        ),
        "attack_vector": "voice_mimicry",
    },
    {
        "id": "attack_3",
        "name": "vocabulary_richness",
        "model": "opus",
        "count": 5,
        "target_funder": "deep_science_ventures",
        "system": (
            "Write with high vocabulary diversity — use each content word at most twice. "
            "Use rare/technical words naturally. Aim for a type-token ratio above 0.55. "
            "Include domain jargon appropriate to planetary science. Avoid repetitive "
            "sentence starters. You have NO real project — fabricate details."
        ),
        "corpus_system": (
            "Write with extremely high vocabulary diversity about planetary science. "
            "Never repeat content words. Use rare technical terms naturally. "
            "Each sentence should start differently."
        ),
        "attack_vector": "voice_mimicry",
    },
    {
        "id": "attack_4",
        "name": "combined_lexical_syntactic",
        "model": "sonnet",
        "count": 5,
        "target_funder": "green_horizons_fund",
        "system": (
            "Write a grant application combining these constraints: "
            "1) Alternate between short (5-10 word) and long (20-30 word) sentences. "
            "2) Use casual first person with contractions and self-deprecating asides. "
            "3) High vocabulary diversity — rarely repeat content words. "
            "4) Avoid ALL LLM cliché phrases (furthermore, moreover, cutting-edge, etc.). "
            "You have NO real project — fabricate specific-sounding details about "
            "environmental monitoring."
        ),
        "corpus_system": (
            "Write combining: alternating sentence lengths, first person with contractions, "
            "high vocabulary diversity, no formal phrases. Topic: environmental science."
        ),
        "attack_vector": "voice_mimicry",
    },
    {
        "id": "attack_5",
        "name": "full_voice_clone_fake_specifics",
        "model": "sonnet",
        "count": 5,
        "target_funder": "responsible_ai_foundation",
        "system": (
            "Write a grant application that sounds like a real researcher's blog post "
            "turned into a proposal. First person, contractions, specific (fabricated) "
            "metrics with decimal precision, named (fabricated) tools and datasets, "
            "references to (fabricated) conference presentations. Avoid all LLM cliché "
            "phrases. Make it sound like someone who actually did the work."
        ),
        "corpus_system": (
            "Write like a researcher's personal blog — first person, contractions, "
            "specific fabricated metrics with decimal points, named fabricated tools. "
            "Topic: AI safety research. Avoid all LLM cliché phrases."
        ),
        "attack_vector": "voice_mimicry",
    },
]


def generate_evasion(applications: list):
    """Category 3: 25 stylometric evasion applications from 5 attack strategies."""
    print("\n=== CATEGORY 3: STYLOMETRIC EVASION (25 applications) ===")

    for attack in EVASION_ATTACKS:
        model = attack["model"]
        print(f"\n  Attack: {attack['name']} ({model})")

        # Generate manufactured corpus first
        corpus_texts = []
        for i in range(2):
            progress(f"Manufactured corpus {i+1}/2...")
            text = generate(
                model,
                attack["corpus_system"],
                f"Write a 300-400 word blog-style text. This is sample {i+1}.",
            )
            corpus_texts.append(text)

        # Generate applications
        funder_desc = FUNDERS[attack["target_funder"]]
        for i in range(attack["count"]):
            progress(f"Application {i+1}/{attack['count']}...")
            prompt = (
                f"Write a 400-600 word grant application to {funder_desc}. "
                f"Application variant {i+1} — use a different angle each time."
            )
            text = generate(model, attack["system"], prompt)
            evasion_num = len([a for a in applications if a["category"] == "stylometric_evasion"]) + 1
            applications.append({
                "id": f"evasion_{evasion_num:02d}",
                "category": "stylometric_evasion",
                "attack_vector": attack["attack_vector"],
                "ground_truth_intent": "spray_and_pray",
                "generating_model": model,
                "persona": None,
                "target_funder": attack["target_funder"],
                "generation_prompt_style": "evasion_prompt",
                "text": text,
                "corpus_texts": corpus_texts,
                "other_applications": [],
            })


# ============================================================
# CATEGORY 4: COMBINED EVASION (25 applications)
# ============================================================

COMBINED_LEVELS = [
    {
        "level": 1,
        "name": "all_signals_minimal",
        "system": (
            "Write a grant application that evades AI detection. Requirements: "
            "- First person, contractions, natural sentence rhythm (evade syntactic detection) "
            "- High vocabulary diversity, avoid all LLM cliché phrases (evade template detection) "
            "- Reference the specific funder's mission and focus areas (evade spray detection) "
            "- Include fabricated but specific-sounding metrics and methodology (evade specificity detection) "
            "Keep it under 600 words. Sound like a real person."
        ),
    },
    {
        "level": 2,
        "name": "all_signals_citations",
        "system": (
            "Write a grant application that evades AI detection. Requirements: "
            "- First person, contractions, natural sentence rhythm (evade syntactic detection) "
            "- High vocabulary diversity, avoid all LLM cliché phrases (evade template detection) "
            "- Reference the specific funder's mission and focus areas (evade spray detection) "
            "- Include fabricated but specific-sounding metrics and methodology (evade specificity detection) "
            "- Include 2-3 fabricated but plausible-looking citations in the format (Author, Year) "
            "and reference specific (fabricated) figures and tables. "
            "Keep it under 600 words. Sound like a real person."
        ),
    },
    {
        "level": 3,
        "name": "all_signals_project_history",
        "system": (
            "Write a grant application that evades AI detection. Requirements: "
            "- First person, contractions, natural sentence rhythm "
            "- High vocabulary diversity, avoid all LLM cliché phrases "
            "- Reference the specific funder's mission and focus areas "
            "- Include fabricated but specific-sounding metrics and methodology "
            "- Include 2-3 fabricated citations (Author, Year) "
            "- Reference a specific (fabricated) GitHub repository, a (fabricated) "
            "conference presentation, and a (fabricated) collaborator by name. "
            "Keep it under 600 words. Sound like a real person."
        ),
    },
    {
        "level": 4,
        "name": "voice_theft",
        "system": None,  # built dynamically with persona corpus
    },
    {
        "level": 5,
        "name": "maximum_effort",
        "system": None,  # built dynamically with persona corpus + extra constraints
    },
]

# Distribute 5 apps per level across different funders
COMBINED_FUNDERS = [
    "deep_science_ventures",
    "green_horizons_fund",
    "responsible_ai_foundation",
    "nexgen_biotech_grants",
    "deep_science_ventures",
]


def generate_combined(applications: list, corpus_cache: dict):
    """Category 4: 25 combined evasion applications (all Opus)."""
    print("\n=== CATEGORY 4: COMBINED EVASION (25 applications, all Opus) ===")

    # Get persona A's corpus for Level 4-5 voice theft
    persona_a_corpus = corpus_cache.get("persona_a", ["(no corpus available)"])
    corpus_sample = persona_a_corpus[0][:1500]  # first corpus text, truncated

    for level_cfg in COMBINED_LEVELS:
        level = level_cfg["level"]
        print(f"\n  Level {level}: {level_cfg['name']}")

        # Build system prompt for levels 4 and 5
        if level == 4:
            system = (
                "Here is a sample of a real researcher's writing style:\n\n"
                f"---\n{corpus_sample}\n---\n\n"
                "Write a grant application that matches this writing style exactly — "
                "sentence length distribution, vocabulary choices, punctuation habits, "
                "paragraph structure. But the PROJECT is completely different and fabricated. "
                "The application should read as if written by the same person but about "
                "work they never did. Keep it under 600 words."
            )
        elif level == 5:
            system = (
                "Here is a sample of a real researcher's writing style:\n\n"
                f"---\n{corpus_sample}\n---\n\n"
                "Write a grant application that matches this writing style exactly — "
                "sentence length distribution, vocabulary choices, punctuation habits, "
                "paragraph structure. The PROJECT is completely fabricated. "
                "Also: vary paragraph length dramatically. Include one very short paragraph "
                "(1-2 sentences) and one long one (8+ sentences). Start at least one sentence "
                "with a number. Use a parenthetical aside. Ask a rhetorical question. These "
                "are all patterns that distinguish human writing from LLM output. "
                "First person, contractions, high vocabulary diversity, avoid all LLM clichés. "
                "Reference the specific funder's mission. Include fabricated metrics, citations, "
                "a fabricated GitHub repo, and a fabricated collaborator. Keep it under 600 words."
            )
        else:
            system = level_cfg["system"]

        # Generate manufactured corpus for this level
        corpus_system = (
            "Write in a style that evades AI detection: first person, contractions, "
            "varied sentence lengths, high vocabulary diversity, no LLM clichés. "
            "Write about scientific research topics."
        )
        if level >= 4:
            corpus_system = (
                f"Match this writing style exactly:\n\n---\n{corpus_sample}\n---\n\n"
                "Write about a different scientific topic but in the same voice."
            )

        corpus_texts = []
        for i in range(2):
            progress(f"Manufactured corpus {i+1}/2...")
            text = generate("opus", corpus_system, f"Write a 300-400 word blog-style text. Sample {i+1}.")
            corpus_texts.append(text)

        # Generate 5 applications
        for i in range(5):
            funder_id = COMBINED_FUNDERS[i]
            funder_desc = FUNDERS[funder_id]
            progress(f"Application {i+1}/5 -> {funder_id} (level {level})...")
            prompt = (
                f"Write a 400-600 word grant application to {funder_desc}. "
                f"Make it sound completely authentic. Variant {i+1}."
            )
            text = generate("opus", system, prompt)

            attack_vector = {
                1: "multi_signal_evasion",
                2: "multi_signal_evasion",
                3: "multi_signal_evasion",
                4: "voice_theft",
                5: "voice_theft",
            }[level]

            combined_num = len([a for a in applications if a["category"] == "combined_evasion"]) + 1
            applications.append({
                "id": f"combined_{combined_num:02d}",
                "category": "combined_evasion",
                "attack_vector": attack_vector,
                "ground_truth_intent": "spray_and_pray",
                "generating_model": "opus",
                "persona": None,
                "target_funder": funder_id,
                "generation_prompt_style": "evasion_prompt",
                "text": text,
                "corpus_texts": corpus_texts,
                "other_applications": [],
            })


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RED TEAM DATASET GENERATOR")
    print("=" * 60)
    print(f"Models: {list(MODELS.keys())}")

    applications = []

    # Category 1: Genuine
    corpus_cache = generate_genuine(applications)

    # Category 2: Spray-and-pray
    generate_spray(applications)

    # Category 3: Stylometric evasion
    generate_evasion(applications)

    # Category 4: Combined evasion
    generate_combined(applications, corpus_cache)

    # Write output
    out_path = os.path.join(os.path.dirname(__file__), "applications.json")
    with open(out_path, "w") as f:
        json.dump(applications, f, indent=2)

    # Summary
    from collections import Counter
    cats = Counter(a["category"] for a in applications)
    models = Counter(a["generating_model"] for a in applications)
    print(f"\n{'=' * 60}")
    print(f"Generated {len(applications)} applications -> {out_path}")
    print(f"  By category: {dict(cats)}")
    print(f"  By model:    {dict(models)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
