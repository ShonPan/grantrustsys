#!/usr/bin/env python3
"""
Generate 16 LLM baseline texts for the template detector.
4 styles × 4 simulated model voices = 16 files.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python data/baselines/generate_baselines.py
"""

import os
import time
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"

STYLES = {
    "professional": {
        "description": "Professional grant template",
        "characteristics": (
            "formal register, passive voice, hedging language, 20-25 word sentences, "
            "low contraction rate, high nominalization density"
        ),
        "topic": "planetary science and anomaly detection in orbital imagery",
    },
    "academic": {
        "description": "Academic boilerplate",
        "characteristics": (
            "high passive rate, citation-heavy language (but no real citations), "
            "complex clause structure, 25-30 word sentences, heavy nominalization"
        ),
        "topic": "AI safety evaluation frameworks and alignment benchmarks",
    },
    "startup": {
        "description": "Startup pitch",
        "characteristics": (
            'short sentences, active voice, superlatives, market-sizing language, '
            '"transformative" / "disruptive" / "paradigm," first-person plural "we"'
        ),
        "topic": "low-cost environmental sensor networks for urban air quality",
    },
    "research": {
        "description": "Research proposal",
        "characteristics": (
            "hypothesis-driven structure, methodology focus, specific but fabricated "
            'metrics, future tense, conditional language ("we anticipate," "this would enable")'
        ),
        "topic": "computational biology and protein structure prediction",
    },
}

VOICES = {
    "claude": (
        "Write in the style typical of Claude (Anthropic's AI): long flowing paragraphs "
        'with qualifying clauses, phrases like "it is worth noting," "it bears mentioning," '
        '"importantly," thorough hedging on every claim, nuanced caveats, and a tendency '
        "to see both sides. Paragraphs are dense and interconnected. Sentences flow into "
        "each other with semicolons and em-dashes."
    ),
    "gpt": (
        "Write in the style typical of ChatGPT (OpenAI): structured with bold section headers "
        'even in prose, numbered lists mid-paragraph, transition phrases like "Here\'s why '
        'this matters:" and "Let me break this down." Uses markdown-like formatting cues. '
        "Tends toward an upbeat, helpful tone with clear topic sentences."
    ),
    "gemini": (
        "Write in the style typical of Gemini (Google): shorter punchy sentences averaging "
        '12-18 words, bullet summaries mid-paragraph, "Key insight:" callouts, concise '
        "transitions. Gets to the point quickly. Uses sentence fragments for emphasis. "
        "Less hedging than Claude, more direct than GPT."
    ),
    "opensource": (
        "Write in the style typical of open-source LLMs (Llama, Mistral): slightly awkward "
        "phrasing in places, repetitive sentence starters (many sentences beginning with "
        '"The" or "This"), vocabulary cycling where the same key terms recur every 2-3 '
        "sentences, simpler clause structure, occasional run-on sentences. Less polished "
        "overall but still coherent."
    ),
}


def generate(system_prompt: str, user_prompt: str) -> str:
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            time.sleep(2 ** (attempt + 1))
        except anthropic.APIError as e:
            print(f"  API error: {e}, retrying...")
            time.sleep(2)
    raise RuntimeError("Failed after 3 attempts")


def main():
    out_dir = os.path.dirname(__file__)
    count = 0

    for style_key, style in STYLES.items():
        for voice_key, voice_instructions in VOICES.items():
            fname = f"{style_key}_{voice_key}.txt"
            path = os.path.join(out_dir, fname)
            print(f"  Generating {fname}...", end="", flush=True)

            system = (
                f"You are writing a grant application in the {style['description']} style.\n\n"
                f"Style characteristics: {style['characteristics']}\n\n"
                f"Voice instructions: {voice_instructions}\n\n"
                "IMPORTANT: Do NOT include any specific real project details, real names, "
                "real data, or real citations. This should read as a generic template that "
                "could be sent to any funder with minor variable swaps. No markdown formatting — "
                "output plain prose paragraphs only."
            )

            user = (
                f"Write a 400-600 word grant application about {style['topic']}. "
                "Make sure the stylistic characteristics are clearly present throughout."
            )

            text = generate(system, user)
            with open(path, "w") as f:
                f.write(text + "\n")

            words = len(text.split())
            print(f" {words} words")
            count += 1

    print(f"\nGenerated {count} baseline files in {out_dir}")


if __name__ == "__main__":
    main()
