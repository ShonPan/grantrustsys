"""
Grant Trust System — End-to-End Pipeline
==========================================
Single-file: Intent Detection Engine + Web Dashboard

Run:
    pip install flask flask-cors numpy scipy scikit-learn
    python grant_trust_pipeline.py
    → http://localhost:5002

No separate files needed.
"""

import os, sys, json, time, threading, traceback
from typing import Dict, List

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS


import re
import json
import hashlib
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Optional: spaCy for richer NLP features
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False

# Optional: textstat for readability metrics
# textstat requires NLTK cmudict corpus which may not be available;
# we fall back to our own syllable counter in those cases.
HAS_TEXTSTAT = False  # use internal approximation for portability


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentScore:
    """
    Output of the intent detection engine for a single application.
    Maps directly to the spec's "Intent score on a spectrum."
    """
    # Overall
    intent_score: float           # 0.0 = automated spray-and-pray, 1.0 = genuine human-in-loop
    label: str                    # "spray_and_pray" | "likely_automated" | "mixed" | "likely_genuine" | "genuine"
    confidence: float             # 0–1 confidence in the classification

    # Component scores (mirrors Xenarch's multi-metric decomposition)
    authorship_score: float       # Stylometric similarity to applicant's own corpus
    template_score: float         # Distance from known LLM output clusters (higher = more divergent = better)
    spray_score: float            # Cross-application uniformity penalty (higher = more spray-like = worse)
    specificity_score: float      # Funder-specific language and genuine engagement signals

    # Feature breakdown (for improvement pathways in component 3)
    feature_detail: Dict          # Raw feature values for explainability
    flags: List[str]              # Human-readable signals that fired
    suggestions: List[str]        # Improvement hints passed to component 3


@dataclass
class ApplicantCorpus:
    """
    Applicant's prior writing corpus — blog posts, READMEs, hackathon submissions.
    These are the "costly signals" from the spec.
    """
    applicant_id: str
    texts: List[str]              # Raw prior writing samples
    metadata: Dict = field(default_factory=dict)   # dates, urls, types


@dataclass
class FunderContext:
    """
    Funder profile used to score specificity.
    Passed in from component 2 (Shon's fit scoring).
    """
    funder_id: str
    mission_keywords: List[str]
    focus_areas: List[str]
    past_recipient_language: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (the "40+ features" from the spec)
# ─────────────────────────────────────────────────────────────────────────────

class StylometricExtractor:
    """
    Extracts stylometric features from text.
    Analogous to Xenarch's chip feature extraction —
    same principle, different signal domain.

    Features are grouped into four families:
      1. Lexical richness       (vocabulary, type-token ratio, rare words)
      2. Syntactic patterns     (sentence structure, punctuation, clause depth)
      3. Surface statistics     (length distributions, readability)
      4. Semantic texture       (hedge language, funder-specific signals, specificity)
    """

    # LLM-generated text tends toward these patterns — used as baseline signals
    LLM_HEDGE_PHRASES = [
        "it is important to note", "in conclusion", "furthermore", "moreover",
        "it is worth mentioning", "as mentioned above", "in summary",
        "i am writing to express", "i am excited to", "deeply passionate about",
        "transformative impact", "cutting-edge", "leverage", "synergize",
        "paradigm shift", "holistic approach", "robust framework",
        "foster collaboration", "drive innovation", "impactful outcomes",
    ]

    # Phrases that suggest genuine engagement with a funder's specific materials
    SPECIFICITY_MARKERS = [
        r"\bin\s+\d{4}\b",                         # references a specific year
        r"\bpage\s+\d+\b",                          # page references
        r"\b(figure|fig\.?)\s+\d+\b",              # figure references
        r"\b(table|tbl\.?)\s+\d+\b",               # table references
        r"\b(appendix|exhibit)\s+[a-z]\b",         # appendix references
        r"\bsection\s+\d+\b",                      # section references
    ]

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def extract(self, text: str) -> np.ndarray:
        """
        Returns a 1-D feature vector for the text.
        Cache is keyed by text hash for efficiency across
        cross-application uniformity checks.
        """
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        vec = np.concatenate([
            self._lexical_features(text),
            self._syntactic_features(text),
            self._surface_features(text),
            self._semantic_texture_features(text),
        ])
        self._cache[key] = vec
        return vec

    # ── 1. Lexical richness (10 features) ────────────────────────────────

    def _lexical_features(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\b[a-z']+\b", text.lower())
        if not tokens:
            return np.zeros(10)

        n = len(tokens)
        vocab = set(tokens)
        freq  = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        counts = list(freq.values())

        ttr          = len(vocab) / n                              # type-token ratio
        hapax        = sum(1 for c in counts if c == 1) / n       # hapax legomena rate
        avg_word_len = np.mean([len(t) for t in tokens])
        std_word_len = np.std([len(t) for t in tokens])

        # Yule's K — lower = more diverse vocabulary
        # Yule's K = 10^4 * (Σ f_i^2 - N) / N^2
        yules_k = (1e4 * (sum(c*c for c in counts) - n)) / (n * n + 1e-8)

        # Brunet's W — another vocabulary richness measure
        brunets_w = n ** (len(vocab) ** -0.165) if len(vocab) > 0 else 0

        # Rare word ratio (words used only 1–2 times)
        rare_ratio   = sum(1 for c in counts if c <= 2) / len(counts)
        freq_entropy = float(entropy([c/n for c in counts]))

        # Long word ratio (> 8 chars, signals academic vs. colloquial register)
        long_word_ratio = sum(1 for t in tokens if len(t) > 8) / n

        # Contraction ratio — LLM text avoids contractions in formal contexts
        contractions = re.findall(r"\b\w+n't\b|\b(i'm|you're|we're|they're|it's|that's)\b",
                                  text.lower())
        contraction_rate = len(contractions) / n

        return np.array([ttr, hapax, avg_word_len, std_word_len,
                         yules_k, brunets_w, rare_ratio, freq_entropy,
                         long_word_ratio, contraction_rate], dtype=np.float32)

    # ── 2. Syntactic patterns (12 features) ──────────────────────────────

    def _syntactic_features(self, text: str) -> np.ndarray:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if not sentences:
            return np.zeros(12)

        sent_lengths  = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        avg_sent_len  = np.mean(sent_lengths)
        std_sent_len  = np.std(sent_lengths)

        # Sentence length entropy — LLM text has unnaturally uniform sentence lengths
        if len(sent_lengths) > 1:
            bins = np.histogram(sent_lengths, bins=min(10, len(sent_lengths)))[0]
            bins = bins[bins > 0]
            sent_len_entropy = float(entropy(bins / bins.sum()))
        else:
            sent_len_entropy = 0.0

        # Punctuation features
        comma_rate   = text.count(",")  / max(len(text.split()), 1)
        dash_rate    = (text.count("—") + text.count("–") + text.count("-")) / max(len(text), 1)
        paren_rate   = text.count("(") / max(len(text.split()), 1)
        semicolon_rate = text.count(";") / max(len(text.split()), 1)

        # Question and exclamation frequency
        question_rate = text.count("?") / max(len(sentences), 1)
        exclaim_rate  = text.count("!") / max(len(sentences), 1)

        # Paragraph structure
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        n_paragraphs = len(paragraphs)
        avg_para_len = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0

        # Subordinate clause indicators
        sub_clause_markers = len(re.findall(
            r"\b(although|however|whereas|despite|while|since|because|unless|if|when)\b",
            text.lower()
        )) / max(len(sentences), 1)

        return np.array([avg_sent_len, std_sent_len, sent_len_entropy,
                         comma_rate, dash_rate, paren_rate, semicolon_rate,
                         question_rate, exclaim_rate, n_paragraphs,
                         avg_para_len, sub_clause_markers], dtype=np.float32)

    # ── 3. Surface statistics (8 features) ───────────────────────────────

    def _surface_features(self, text: str) -> np.ndarray:
        words = text.split()
        n_words = len(words)
        n_chars = len(text)
        n_sents = max(1, len(re.split(r"[.!?]+", text)))

        # Readability (uses textstat if available, else Flesch approximation)
        if HAS_TEXTSTAT:
            flesch    = textstat.flesch_reading_ease(text) / 100.0
            fk_grade  = textstat.flesch_kincaid_grade(text) / 20.0
            gunning   = textstat.gunning_fog(text) / 20.0
        else:
            # Approximate Flesch: 206.835 - 1.015*(words/sents) - 84.6*(syllables/words)
            syllables = sum(self._count_syllables(w) for w in words)
            flesch   = max(0, min(1, (206.835 - 1.015*(n_words/n_sents)
                                     - 84.6*(syllables/max(n_words,1))) / 100.0))
            fk_grade = min(1, 0.39*(n_words/n_sents) + 11.8*(syllables/max(n_words,1)) - 15.59) / 20.0
            gunning  = 0.5 * ((n_words/n_sents) + 100 * sum(
                1 for w in words if self._count_syllables(w) >= 3
            ) / max(n_words,1)) / 20.0

        # Text length (log-normalised)
        log_length = np.log1p(n_words) / 10.0

        # Average characters per word
        avg_char_per_word = n_chars / max(n_words, 1) / 10.0

        # Digit presence ratio
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(n_chars, 1)

        # Uppercase ratio (ACRONYMS, SHOUTING — can signal copy-paste)
        upper_ratio = sum(1 for c in text if c.isupper()) / max(n_chars, 1)

        return np.array([flesch, fk_grade, gunning, log_length,
                         avg_char_per_word, digit_ratio, upper_ratio,
                         float(n_sents) / max(n_words, 1)], dtype=np.float32)

    def _count_syllables(self, word: str) -> int:
        """Naive syllable counter for readability approximation."""
        word = word.lower().strip(".,!?;:")
        if not word:
            return 0
        count = len(re.findall(r"[aeiou]+", word))
        if word.endswith("e"):
            count -= 1
        return max(1, count)

    # ── 4. Semantic texture (12 features) ────────────────────────────────

    def _semantic_texture_features(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        words      = re.findall(r"\b\w+\b", text_lower)
        n_words    = max(len(words), 1)

        # LLM cliché density — key signal from spec
        llm_phrase_hits = sum(
            1 for phrase in self.LLM_HEDGE_PHRASES
            if phrase in text_lower
        )
        llm_density = llm_phrase_hits / len(self.LLM_HEDGE_PHRASES)

        # Specificity markers (references, citations, page numbers)
        spec_hits = sum(
            1 for pattern in self.SPECIFICITY_MARKERS
            if re.search(pattern, text_lower)
        )
        specificity_marker_rate = spec_hits / len(self.SPECIFICITY_MARKERS)

        # First-person singular vs. plural ratio
        # Genuine applicants use "I built…" not "we believe…" (unless team)
        first_singular  = len(re.findall(r"\b(i|i've|i'm|i'd|i'll|my|myself)\b", text_lower))
        first_plural    = len(re.findall(r"\b(we|we've|we're|we'd|our|ourselves)\b", text_lower))
        fp_ratio        = first_singular / (first_singular + first_plural + 1)

        # Hedging language density
        hedge_words = re.findall(
            r"\b(perhaps|maybe|might|could|may|seems|appears|suggest|indicate|likely)\b",
            text_lower
        )
        hedge_rate = len(hedge_words) / n_words

        # Active vs. passive voice approximation
        passive_hits = len(re.findall(
            r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", text_lower
        ))
        passive_rate = passive_hits / n_words

        # Named entity density (proper nouns, specific places/people — genuine signal)
        if HAS_SPACY:
            doc = _nlp(text[:5000])   # truncate for speed
            ne_density = len(doc.ents) / max(len(doc), 1)
        else:
            # Approximate: capitalized non-sentence-start tokens
            all_tokens   = re.findall(r"\b[A-Z][a-z]+\b", text)
            ne_density   = len(all_tokens) / n_words

        # Numeric specificity (concrete numbers suggest real work)
        numeric_hits = re.findall(r"\b\d+(?:\.\d+)?(?:%|x|\b)", text)
        numeric_density = len(numeric_hits) / n_words

        # Technical vocabulary depth (domain jargon — harder to fake)
        tech_patterns = [
            r"\b\w+(?:tion|ment|ance|ence|ity|ous|ive|ize|ise)\b",   # nominalizations
            r"\b[A-Z]{2,}\b",                                          # acronyms
        ]
        tech_count = sum(len(re.findall(p, text)) for p in tech_patterns)
        tech_density = tech_count / n_words

        # Temporal specificity (dates, durations — "in Q3 2024 we ran…")
        temporal_hits = re.findall(
            r"\b(\d{4}|Q[1-4]|january|february|march|april|may|june|july|august"
            r"|september|october|november|december|last\s+\w+|next\s+\w+)\b",
            text_lower
        )
        temporal_density = len(temporal_hits) / n_words

        # URL / citation presence
        url_hits = len(re.findall(r"https?://\S+|github\.com/\S+|arxiv\.org/\S+", text))
        citation_hits = len(re.findall(r"\[\d+\]|\(\w+,\s+\d{4}\)", text))
        evidence_rate = (url_hits + citation_hits) / n_words

        return np.array([
            llm_density, specificity_marker_rate, fp_ratio,
            hedge_rate, passive_rate, ne_density, numeric_density,
            tech_density, temporal_density, evidence_rate,
            float(url_hits) / n_words,
            float(citation_hits) / n_words,
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# AUTHORSHIP SCORER
# Xenarch analogue: "compare vs. applicant prior corpus"
# ─────────────────────────────────────────────────────────────────────────────

class AuthorshipScorer:
    """
    Computes stylometric similarity between an application
    and the applicant's known prior writing.

    High authorship score = application reads like the person's own work.
    Low authorship score  = high divergence from their voice.
    """

    def __init__(self, extractor: StylometricExtractor):
        self.extractor = extractor

    def build_corpus_profile(self, corpus: ApplicantCorpus) -> Optional[np.ndarray]:
        """
        Returns the mean feature vector across the applicant's prior texts.
        Returns None if corpus is empty.
        """
        if not corpus.texts:
            return None
        vecs = [self.extractor.extract(t) for t in corpus.texts if len(t.split()) > 50]
        if not vecs:
            return None
        return np.stack(vecs).mean(axis=0)

    def score(self,
              application_text: str,
              corpus: Optional[ApplicantCorpus]) -> Tuple[float, Dict]:
        """
        Returns (authorship_score, detail_dict).
        authorship_score: 0–1, where 1 = clearly the applicant's own voice.
        """
        if corpus is None or not corpus.texts:
            return 0.5, {"reason": "no_corpus", "similarity": None}

        app_vec    = self.extractor.extract(application_text)
        corpus_vec = self.build_corpus_profile(corpus)

        if corpus_vec is None:
            return 0.5, {"reason": "corpus_too_short", "similarity": None}

        # Cosine similarity between application and corpus profile
        similarity = 1.0 - cosine(app_vec, corpus_vec)
        similarity = float(np.clip(similarity, 0, 1))

        # Per-family divergence breakdown
        # Feature vector layout: [lexical×10, syntactic×12, surface×8, semantic×12]
        slices = {"lexical": (0,10), "syntactic": (10,22), "surface": (22,30), "semantic": (30,42)}
        family_divergence = {}
        for name, (a, b) in slices.items():
            av = app_vec[a:b]
            cv = corpus_vec[a:b]
            family_divergence[name] = float(np.mean(np.abs(av - cv)))

        # Authorship score: high cosine similarity → high score
        # Apply a soft penalty for high semantic divergence (LLM indicators)
        semantic_penalty = family_divergence.get("semantic", 0.0) * 0.3
        authorship_score = float(np.clip(similarity - semantic_penalty, 0, 1))

        return authorship_score, {
            "cosine_similarity":    round(similarity, 4),
            "family_divergence":    {k: round(v, 4) for k, v in family_divergence.items()},
            "authorship_score":     round(authorship_score, 4),
            "corpus_size":          len(corpus.texts),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE DETECTOR
# Xenarch analogue: "DBSCAN clustering vs. LLM output baselines"
# ─────────────────────────────────────────────────────────────────────────────

class TemplateDetector:
    """
    Clusters the application against known LLM output patterns using DBSCAN.
    Applications that cluster tightly with LLM baselines score low.
    Applications with distinctive stylometric signatures score high.

    The LLM baseline corpus is built from known patterns;
    in production this would be replaced with a real corpus.
    """

    # Synthetic feature-space representatives of LLM grant writing patterns.
    # These approximate the stylometric signature of GPT/Claude/Gemini
    # grant application outputs based on known characteristics:
    # - high sentence length uniformity (low std_sent_len, mid avg)
    # - high LLM cliché density
    # - low contraction rate
    # - high passive rate
    # - low hapax rate (repetitive vocabulary)
    _LLM_BASELINE_PROFILES: List[Dict] = [
        # "Professional grant template" style (most common spray-and-pray)
        {"avg_sent_len": 22, "std_sent_len": 4, "llm_density": 0.7, "passive_rate": 0.08,
         "contraction_rate": 0.002, "ttr": 0.45, "hedge_rate": 0.02, "numeric_density": 0.01},
        # "Academic boilerplate" style
        {"avg_sent_len": 28, "std_sent_len": 5, "llm_density": 0.6, "passive_rate": 0.12,
         "contraction_rate": 0.001, "ttr": 0.42, "hedge_rate": 0.03, "numeric_density": 0.005},
        # "Startup pitch" LLM style
        {"avg_sent_len": 18, "std_sent_len": 3, "llm_density": 0.8, "passive_rate": 0.05,
         "contraction_rate": 0.003, "ttr": 0.48, "hedge_rate": 0.01, "numeric_density": 0.02},
        # "Research proposal" LLM style
        {"avg_sent_len": 25, "std_sent_len": 6, "llm_density": 0.5, "passive_rate": 0.15,
         "contraction_rate": 0.001, "ttr": 0.40, "hedge_rate": 0.04, "numeric_density": 0.03},
    ]

    # Feature indices for the condensed template-detection subspace
    # (subset of the full 42-feature vector focused on template-sensitive signals)
    _TEMPLATE_FEATURE_NAMES = [
        "avg_sent_len", "std_sent_len", "sent_len_entropy",
        "llm_density", "passive_rate", "contraction_rate",
        "ttr", "hedge_rate", "numeric_density", "specificity_marker_rate"
    ]

    def __init__(self, extractor: StylometricExtractor):
        self.extractor = extractor
        self.scaler    = StandardScaler()
        self._fitted   = False

    def _profile_to_vec(self, profile: Dict) -> np.ndarray:
        """Convert a synthetic LLM profile dict to a feature vector."""
        return np.array([
            profile.get("avg_sent_len", 20) / 30.0,
            profile.get("std_sent_len",  5)  / 10.0,
            profile.get("sent_len_entropy", 1.5) / 3.0,
            profile.get("llm_density", 0.5),
            profile.get("passive_rate", 0.08),
            profile.get("contraction_rate", 0.002) * 100,
            profile.get("ttr", 0.45),
            profile.get("hedge_rate", 0.02) * 10,
            profile.get("numeric_density", 0.01) * 10,
            profile.get("specificity_marker_rate", 0.1),
        ], dtype=np.float32)

    def _extract_template_subspace(self, full_vec: np.ndarray) -> np.ndarray:
        """
        Extract the template-sensitive subspace from the full feature vector.
        Feature index mapping into the 42-vector:
          lexical[0]=ttr, lexical[1]=hapax, lexical[9]=contraction_rate
          syntactic[0]=avg_sent_len, syntactic[1]=std_sent_len, syntactic[2]=sent_len_entropy
          syntactic[11]=sub_clause
          surface[0]=flesch, surface[1]=fk_grade
          semantic[0]=llm_density, semantic[3]=hedge_rate, semantic[4]=passive_rate
          semantic[6]=numeric_density, semantic[1]=specificity_marker_rate
        """
        return np.array([
            full_vec[10],  # avg_sent_len
            full_vec[11],  # std_sent_len
            full_vec[12],  # sent_len_entropy
            full_vec[30],  # llm_density
            full_vec[34],  # passive_rate
            full_vec[9],   # contraction_rate
            full_vec[0],   # ttr
            full_vec[33],  # hedge_rate
            full_vec[36],  # numeric_density
            full_vec[31],  # specificity_marker_rate
        ], dtype=np.float32)

    def score(self, application_text: str,
              additional_corpus: Optional[List[str]] = None) -> Tuple[float, Dict]:
        """
        Returns (template_score, detail_dict).
        template_score: 0–1, where 1 = maximally distinctive from LLM baselines.

        DBSCAN logic mirrors Xenarch's chip clustering:
          - Build a feature space with LLM baseline profiles + the application
          - If the application clusters with LLM baselines → low score
          - If it's an outlier (label = -1 in DBSCAN) → high score
        """
        full_vec = self.extractor.extract(application_text)
        app_sub  = self._extract_template_subspace(full_vec)

        # Build the baseline corpus in template subspace
        baseline_vecs = [self._profile_to_vec(p) for p in self._LLM_BASELINE_PROFILES]
        if additional_corpus:
            for text in additional_corpus[:20]:  # cap for performance
                v = self.extractor.extract(text)
                baseline_vecs.append(self._extract_template_subspace(v))

        # Fit scaler on baseline + application together
        all_vecs   = np.stack(baseline_vecs + [app_sub])
        scaled     = self.scaler.fit_transform(all_vecs)
        app_scaled = scaled[-1]

        # DBSCAN clustering (mirrors Xenarch's cluster detection)
        db = DBSCAN(eps=1.5, min_samples=2, metric="euclidean")
        labels = db.fit_predict(scaled[:-1])  # cluster baselines only

        # Distance from application to each baseline cluster center
        unique_labels = set(labels) - {-1}
        min_dist_to_cluster = float("inf")
        closest_cluster_label = None

        for lbl in unique_labels:
            cluster_vecs  = scaled[:-1][labels == lbl]
            cluster_center = cluster_vecs.mean(axis=0)
            dist = float(np.linalg.norm(app_scaled - cluster_center))
            if dist < min_dist_to_cluster:
                min_dist_to_cluster = dist
                closest_cluster_label = int(lbl)

        # Distance to nearest baseline point
        dists_to_baselines = [
            float(np.linalg.norm(app_scaled - bv))
            for bv in scaled[:-1]
        ]
        min_dist_to_any_baseline = min(dists_to_baselines)
        avg_dist_to_baselines    = float(np.mean(dists_to_baselines))

        # LLM density of the application itself (from semantic features)
        llm_density_raw = float(full_vec[30])

        # Template score: high distance from baselines + low intrinsic LLM density
        # Sigmoid-normalise the distance
        raw_template_score = float(np.tanh(min_dist_to_any_baseline / 3.0))
        llm_penalty        = llm_density_raw * 0.4
        template_score     = float(np.clip(raw_template_score - llm_penalty, 0, 1))

        return template_score, {
            "min_dist_to_baseline":     round(min_dist_to_any_baseline, 4),
            "avg_dist_to_baselines":    round(avg_dist_to_baselines, 4),
            "closest_cluster":          closest_cluster_label,
            "llm_density_raw":          round(llm_density_raw, 4),
            "raw_template_score":       round(raw_template_score, 4),
            "llm_penalty":              round(llm_penalty, 4),
            "template_score":           round(template_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SPRAY DETECTOR
# Xenarch analogue: cross-chip uniformity check
# ─────────────────────────────────────────────────────────────────────────────

class SprayDetector:
    """
    Detects spray-and-pray patterns across multiple applications
    from the same applicant.

    The spec lists three indicators:
      1. Stylometric uniformity (same template, light variable swaps)
      2. Low specificity to different funders' missions
      3. Generic language patterns

    Analogous to Xenarch's "same chip, different file" detection —
    here we're looking for the stylometric equivalent of tiled imagery.
    """

    def __init__(self, extractor: StylometricExtractor):
        self.extractor = extractor

    def score(self, application_text: str,
              other_applications: List[str],
              funder_context: Optional[FunderContext] = None) -> Tuple[float, Dict]:
        """
        Returns (spray_score, detail_dict).
        spray_score: 0–1, where 0 = unique/genuine, 1 = copy-paste spray.

        If other_applications is empty, falls back to intrinsic indicators only.
        """
        detail: Dict = {}

        # ── Intrinsic spray indicators (no other applications needed) ──

        text_lower = application_text.lower()

        # Generic opener patterns — a hallmark of spray-and-pray
        generic_opener_patterns = [
            r"^i am (writing|applying) to (express|request|seek)",
            r"^dear (grant committee|selection committee|review panel|funder)",
            r"^(my name is|i am a) .{0,50} (applying|interested in|excited about)",
            r"^i (have long|have always) been (passionate|interested|committed)",
            r"^this (proposal|application|project) (seeks|aims|intends) to",
        ]
        opener = application_text[:300].lower()
        generic_opener_score = sum(
            1 for p in generic_opener_patterns if re.search(p, opener)
        ) / len(generic_opener_patterns)

        # Funder specificity (does the application mention the funder by name or
        # reference their specific materials?)
        funder_specificity = 0.0
        if funder_context:
            mission_hits = sum(
                1 for kw in funder_context.mission_keywords
                if kw.lower() in text_lower
            )
            funder_specificity = min(1.0, mission_hits / max(len(funder_context.mission_keywords), 1))
            detail["funder_mission_keyword_hits"] = mission_hits
            detail["funder_mission_keywords_total"] = len(funder_context.mission_keywords)
        else:
            funder_specificity = 0.5   # unknown — neutral

        low_specificity_penalty = 1.0 - funder_specificity

        # ── Cross-application uniformity (if other applications provided) ──

        uniformity_score = 0.0
        pairwise_sims    = []

        if other_applications:
            app_vec   = self.extractor.extract(application_text)
            other_vecs = [self.extractor.extract(t) for t in other_applications
                          if len(t.split()) > 50]
            if other_vecs:
                for ov in other_vecs:
                    sim = 1.0 - cosine(app_vec, ov)
                    pairwise_sims.append(float(np.clip(sim, 0, 1)))

                avg_sim      = float(np.mean(pairwise_sims))
                max_sim      = float(np.max(pairwise_sims))
                # High average similarity across applications = spray signal
                uniformity_score = avg_sim

                detail["cross_app_avg_similarity"] = round(avg_sim, 4)
                detail["cross_app_max_similarity"] = round(max_sim, 4)
                detail["n_other_apps_compared"]    = len(pairwise_sims)

        # ── Combine into a spray score ──
        if pairwise_sims:
            # Full signal: cross-app uniformity weighted heavily
            spray_score = float(np.clip(
                0.40 * uniformity_score +
                0.35 * low_specificity_penalty +
                0.25 * generic_opener_score,
                0, 1
            ))
        else:
            # Intrinsic only: opener + specificity
            spray_score = float(np.clip(
                0.55 * low_specificity_penalty +
                0.45 * generic_opener_score,
                0, 1
            ))

        detail.update({
            "generic_opener_score":      round(generic_opener_score, 4),
            "funder_specificity":        round(funder_specificity, 4),
            "low_specificity_penalty":   round(low_specificity_penalty, 4),
            "uniformity_score":          round(uniformity_score, 4),
            "spray_score":               round(spray_score, 4),
        })

        return spray_score, detail


# ─────────────────────────────────────────────────────────────────────────────
# SPECIFICITY SCORER
# Genuine engagement signals
# ─────────────────────────────────────────────────────────────────────────────

class SpecificityScorer:
    """
    Detects genuine engagement signals:
    - Specific references showing they actually read the funder's materials
    - Revision patterns suggesting iteration rather than one-shot generation
    - Concrete details about the applicant's own work

    This is the positive-signal counterpart to the spray/template detectors.
    """

    def __init__(self, extractor: StylometricExtractor):
        self.extractor = extractor

    def score(self, application_text: str,
              funder_context: Optional[FunderContext] = None,
              corpus: Optional[ApplicantCorpus] = None) -> Tuple[float, Dict]:
        """Returns (specificity_score, detail_dict)."""
        text_lower = application_text.lower()
        full_vec   = self.extractor.extract(application_text)
        detail: Dict = {}

        # Numeric specificity from semantic features (index 36)
        numeric_density = float(full_vec[36])
        detail["numeric_density"] = round(numeric_density, 4)

        # Evidence presence (URLs, citations — index 40, 41)
        evidence_rate = float(full_vec[40] + full_vec[41])
        detail["evidence_rate"] = round(evidence_rate, 4)

        # Named entity density (index 35)
        ne_density = float(full_vec[35])
        detail["ne_density"] = round(ne_density, 4)

        # Funder-specific language
        funder_signal = 0.0
        if funder_context:
            # Check for references to specific past recipients, focus areas
            focus_hits = sum(
                1 for area in funder_context.focus_areas
                if area.lower() in text_lower
            )
            past_hits = sum(
                1 for phrase in funder_context.past_recipient_language
                if phrase.lower() in text_lower
            )
            funder_signal = min(1.0, (focus_hits + past_hits * 2) /
                                max(len(funder_context.focus_areas) + 1, 1))
            detail["funder_focus_area_hits"] = focus_hits
            detail["past_recipient_language_hits"] = past_hits
        else:
            funder_signal = 0.4   # neutral if no context

        # Personal project references — corpus comparison
        # If the application references specific projects from their corpus, that's genuine
        corpus_reference_score = 0.0
        if corpus and corpus.texts:
            # Extract key nouns from corpus and check if application references them
            all_corpus_text = " ".join(corpus.texts).lower()
            # Simple heuristic: look for 4+ letter tokens that appear in both
            corpus_tokens = set(re.findall(r"\b[a-z]{4,}\b", all_corpus_text))
            app_tokens    = set(re.findall(r"\b[a-z]{4,}\b", text_lower))
            overlap = corpus_tokens & app_tokens
            # Penalise stopwords
            stopwords = {"that", "this", "with", "from", "have", "been", "they",
                         "will", "also", "more", "than", "some", "when", "what"}
            overlap -= stopwords
            corpus_reference_score = min(1.0, len(overlap) / max(len(corpus_tokens), 100))
            detail["corpus_vocabulary_overlap"] = round(corpus_reference_score, 4)

        # Combine
        specificity_score = float(np.clip(
            0.25 * numeric_density * 5 +    # scale numeric density
            0.20 * evidence_rate * 20 +     # scale evidence rate
            0.15 * ne_density * 5 +         # scale ne density
            0.25 * funder_signal +
            0.15 * corpus_reference_score,
            0, 1
        ))
        detail["specificity_score"] = round(specificity_score, 4)

        return specificity_score, detail


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION ENGINE  (top-level, mirrors Xenarch's run_analysis)
# ─────────────────────────────────────────────────────────────────────────────

class IntentDetectionEngine:
    """
    Top-level engine. Combines all four scorers into a final IntentScore.

    Scoring weights (mirrors Xenarch's combined score formula):
      S = 0.30 * authorship
        + 0.25 * template_score      (inverted: high distance = good)
        + 0.25 * (1 - spray_score)   (inverted: low spray = good)
        + 0.20 * specificity_score

    Thresholds for label assignment:
      0.0–0.25  → spray_and_pray
      0.25–0.45 → likely_automated
      0.45–0.60 → mixed
      0.60–0.80 → likely_genuine
      0.80–1.0  → genuine
    """

    LABEL_THRESHOLDS = [
        (0.25, "spray_and_pray"),
        (0.45, "likely_automated"),
        (0.60, "mixed"),
        (0.80, "likely_genuine"),
        (1.01, "genuine"),
    ]

    # Weights for combined intent score
    WEIGHTS = {
        "authorship":   0.30,
        "template":     0.25,
        "anti_spray":   0.25,
        "specificity":  0.20,
    }

    def __init__(self):
        self.extractor   = StylometricExtractor()
        self.authorship  = AuthorshipScorer(self.extractor)
        self.template    = TemplateDetector(self.extractor)
        self.spray       = SprayDetector(self.extractor)
        self.specificity = SpecificityScorer(self.extractor)

    def analyze(self,
                application_text: str,
                corpus: Optional[ApplicantCorpus]          = None,
                other_applications: Optional[List[str]]    = None,
                funder_context: Optional[FunderContext]     = None,
                llm_baseline_corpus: Optional[List[str]]   = None) -> IntentScore:
        """
        Main entry point. Returns a fully populated IntentScore.

        Parameters
        ----------
        application_text      The grant application to score.
        corpus                Applicant's prior writing (blog posts, READMEs, etc.).
        other_applications    Other applications by the same person (for spray detection).
        funder_context        Funder profile for specificity scoring.
        llm_baseline_corpus   Additional LLM-generated texts for template calibration.
        """

        # ── Run all four scorers ──────────────────────────────────────────

        auth_score, auth_detail = self.authorship.score(
            application_text, corpus
        )

        tmpl_score, tmpl_detail = self.template.score(
            application_text, llm_baseline_corpus
        )

        spray_score, spray_detail = self.spray.score(
            application_text,
            other_applications or [],
            funder_context
        )

        spec_score, spec_detail = self.specificity.score(
            application_text, funder_context, corpus
        )

        # ── Combine scores ────────────────────────────────────────────────

        anti_spray = 1.0 - spray_score

        intent_score = float(np.clip(
            self.WEIGHTS["authorship"]  * auth_score  +
            self.WEIGHTS["template"]    * tmpl_score  +
            self.WEIGHTS["anti_spray"]  * anti_spray  +
            self.WEIGHTS["specificity"] * spec_score,
            0, 1
        ))

        # ── Label assignment ──────────────────────────────────────────────

        label = "genuine"
        for threshold, lbl in self.LABEL_THRESHOLDS:
            if intent_score < threshold:
                label = lbl
                break

        # ── Confidence: how certain are we in the label? ─────────────────
        # High confidence when scores are consistent; lower when mixed signals

        component_scores = np.array([auth_score, tmpl_score, anti_spray, spec_score])
        score_std    = float(np.std(component_scores))
        confidence   = float(np.clip(1.0 - score_std * 1.5, 0.3, 0.99))

        # ── Generate human-readable flags ─────────────────────────────────

        flags = self._generate_flags(
            auth_score, tmpl_score, spray_score, spec_score,
            auth_detail, tmpl_detail, spray_detail, spec_detail
        )

        # ── Generate improvement suggestions (for component 3) ───────────

        suggestions = self._generate_suggestions(
            label, auth_score, tmpl_score, spray_score, spec_score,
            corpus, funder_context
        )

        # ── Assemble full feature detail ──────────────────────────────────

        feature_detail = {
            "component_scores": {
                "authorship":        round(auth_score, 4),
                "template_distance": round(tmpl_score, 4),
                "anti_spray":        round(anti_spray, 4),
                "specificity":       round(spec_score, 4),
            },
            "authorship_detail":   auth_detail,
            "template_detail":     tmpl_detail,
            "spray_detail":        spray_detail,
            "specificity_detail":  spec_detail,
            "weights_used":        self.WEIGHTS,
        }

        return IntentScore(
            intent_score      = round(intent_score, 4),
            label             = label,
            confidence        = round(confidence, 4),
            authorship_score  = round(auth_score, 4),
            template_score    = round(tmpl_score, 4),
            spray_score       = round(spray_score, 4),
            specificity_score = round(spec_score, 4),
            feature_detail    = feature_detail,
            flags             = flags,
            suggestions       = suggestions,
        )

    def _generate_flags(self, auth, tmpl, spray, spec,
                        auth_d, tmpl_d, spray_d, spec_d) -> List[str]:
        flags = []

        if auth < 0.35:
            flags.append("HIGH_AUTHORSHIP_DIVERGENCE: Application reads differently from applicant's known writing.")
        if auth > 0.70:
            flags.append("STRONG_AUTHORSHIP_MATCH: Application is stylistically consistent with applicant's corpus.")

        if tmpl < 0.30:
            flags.append("TEMPLATE_CLUSTER_MATCH: Application stylometrically similar to known LLM output patterns.")
        if tmpl_d.get("llm_density_raw", 0) > 0.5:
            flags.append("HIGH_LLM_PHRASE_DENSITY: Multiple LLM cliché phrases detected.")

        if spray > 0.65:
            flags.append("SPRAY_PATTERN_DETECTED: High cross-application uniformity or low funder specificity.")
        if spray_d.get("generic_opener_score", 0) > 0.4:
            flags.append("GENERIC_OPENER: Application begins with a templated introduction.")
        if spray_d.get("cross_app_avg_similarity", 0) > 0.75:
            flags.append("COPY_PASTE_RISK: Very high similarity to other applications by this applicant.")

        if spec > 0.65:
            flags.append("HIGH_SPECIFICITY: Application contains concrete references and funder-specific language.")
        if spec_d.get("funder_focus_area_hits", 0) == 0 and spec_d.get("funder_focus_area_hits") is not None:
            flags.append("NO_FUNDER_ALIGNMENT: Application doesn't reference this funder's focus areas.")

        return flags

    def _generate_suggestions(self, label, auth, tmpl, spray, spec,
                               corpus, funder_context) -> List[str]:
        """
        Generates suggestions passed to component 3 (Improvement Pathways).
        Mirrors the spec's "ladder, not wall" principle.
        """
        suggestions = []

        if label in ("spray_and_pray", "likely_automated"):
            if tmpl < 0.40:
                suggestions.append(
                    "VOICE: Your application reads as heavily AI-generated. "
                    "Rewrite the opening paragraph in your own words — describe "
                    "specifically what you've built or discovered, not what you hope to do."
                )
            if spray > 0.60:
                suggestions.append(
                    "SPECIFICITY: This application appears to be a generic template. "
                    "Add at least one paragraph referencing a specific past project "
                    "you shipped and what you learned from it."
                )

        if auth < 0.40 and corpus and corpus.texts:
            suggestions.append(
                "VOICE_MATCH: Your application doesn't sound like your other writing. "
                "Read your README or blog post before revising — bring that voice into "
                "the proposal."
            )
        elif auth < 0.40:
            suggestions.append(
                "CORPUS: You have no prior writing on file. Publishing even one technical "
                "blog post or README will significantly strengthen your credibility signal."
            )

        if funder_context and spec < 0.50:
            focus = ", ".join(funder_context.focus_areas[:3])
            suggestions.append(
                f"FUNDER_FIT: Your application doesn't engage with this funder's stated "
                f"focus areas ({focus}). Revise to show how your work specifically addresses "
                f"their mission."
            )

        if not suggestions:
            suggestions.append(
                "STRONG: Your application shows genuine engagement. "
                "Consider adding one more concrete data point or citation to maximize "
                "your fit score."
            )

        return suggestions

    def batch_analyze(self,
                      applications: List[Dict]) -> List[IntentScore]:
        """
        Analyze a batch of applications.
        Each dict should have keys: text, corpus, other_applications, funder_context.
        Shared other_applications are computed automatically within the batch
        when applicant_id is provided.
        """
        # Group by applicant_id to build cross-application sets
        by_applicant: Dict[str, List[int]] = {}
        for i, app in enumerate(applications):
            aid = app.get("applicant_id", f"unknown_{i}")
            by_applicant.setdefault(aid, []).append(i)

        results = []
        for i, app in enumerate(applications):
            aid = app.get("applicant_id", f"unknown_{i}")
            # Other apps by same applicant
            sibling_indices = [j for j in by_applicant.get(aid, []) if j != i]
            other_apps = [applications[j]["text"] for j in sibling_indices]

            result = self.analyze(
                application_text    = app["text"],
                corpus              = app.get("corpus"),
                other_applications  = other_apps or app.get("other_applications", []),
                funder_context      = app.get("funder_context"),
                llm_baseline_corpus = app.get("llm_baseline_corpus"),
            )
            results.append(result)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  (mirrors the spec's three-applicant demo flow)
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# WEB DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS


app   = Flask(__name__)
CORS(app)
ENGINE = IntentDetectionEngine()

# ── Demo data ─────────────────────────────────────────────────────────────────

DEMO_FUNDERS = {
    "deep_science_ventures": FunderContext(
        funder_id="deep_science_ventures",
        mission_keywords=["frontier science","planetary","technosignature",
                          "anomaly detection","remote sensing","ML for science",
                          "open science","reproducible"],
        focus_areas=["astrobiology","planetary science","AI/ML","open science"],
        past_recipient_language=["validated on real orbital data","open-source pipeline",
                                 "reproducible methodology","NASA","ESA"]
    ),
    "civic_tech_fund": FunderContext(
        funder_id="civic_tech_fund",
        mission_keywords=["community","urban","public good","civic","open data",
                          "environmental","sustainability","infrastructure"],
        focus_areas=["civic technology","environmental monitoring","open data","urban systems"],
        past_recipient_language=["deployed in production","real users","community partnership"]
    ),
    "ai_safety_institute": FunderContext(
        funder_id="ai_safety_institute",
        mission_keywords=["alignment","safety","interpretability","robustness",
                          "evaluation","red-teaming","oversight"],
        focus_areas=["AI safety","alignment research","interpretability","governance"],
        past_recipient_language=["empirical safety research","published benchmarks","open weights"]
    ),
}

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

FUNDER_LABELS = {
    "deep_science_ventures": "Deep Science Ventures",
    "civic_tech_fund": "Civic Tech Fund",
    "ai_safety_institute": "AI Safety Institute",
}

# ── Frontend HTML ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# WEB DASHBOARD  —  Grant Trust System End-to-End Pipeline
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, json, traceback
from typing import Dict, List

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app    = Flask(__name__)
CORS(app)
ENGINE = IntentDetectionEngine()

# ── Demo / funder data ────────────────────────────────────────────────────────

DEMO_FUNDERS = {
    "deep_science_ventures": FunderContext(
        funder_id="deep_science_ventures",
        mission_keywords=["frontier science","planetary","technosignature",
                          "anomaly detection","remote sensing","ML for science",
                          "open science","reproducible"],
        focus_areas=["astrobiology","planetary science","AI/ML","open science"],
        past_recipient_language=["validated on real orbital data","open-source pipeline",
                                 "reproducible methodology","NASA","ESA"]
    ),
    "civic_tech_fund": FunderContext(
        funder_id="civic_tech_fund",
        mission_keywords=["community","urban","public good","civic","open data",
                          "environmental","sustainability","infrastructure"],
        focus_areas=["civic technology","environmental monitoring","open data","urban systems"],
        past_recipient_language=["deployed in production","real users","community partnership"]
    ),
    "ai_safety_institute": FunderContext(
        funder_id="ai_safety_institute",
        mission_keywords=["alignment","safety","interpretability","robustness",
                          "evaluation","red-teaming","oversight"],
        focus_areas=["AI safety","alignment research","interpretability","governance"],
        past_recipient_language=["empirical safety research","published benchmarks","open weights"]
    ),
}

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

# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>GRANT TRUST SYSTEM · End-to-End Pipeline</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#04060d;--surf:#090e1a;--panel:#0d1526;--brd:#1a2744;
  --acc:#00e5ff;--acc2:#ff4f00;--grn:#00d97e;--amb:#ffb800;--red:#ff4444;
  --dim:#3a5080;--txt:#c8daf5;--txtlo:#4a6080;
  --mono:'Share Tech Mono',monospace;--sans:'Barlow Condensed',sans-serif;
  --glow:0 0 16px rgba(0,229,255,.35);
}
html,body{height:100%;background:var(--bg);color:var(--txt);font-family:var(--sans)}
body::before{content:'';position:fixed;inset:0;z-index:9999;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px)}

/* ── layout ── */
.shell{max-width:1440px;margin:0 auto;padding:0 24px 80px}

/* ── header ── */
header{display:flex;align-items:center;gap:16px;padding:24px 0 18px;
  border-bottom:1px solid var(--brd);margin-bottom:24px}
.logo-mark{width:42px;height:42px;border:2px solid var(--acc);border-radius:4px;
  display:grid;place-items:center;box-shadow:var(--glow);position:relative;flex-shrink:0}
.logo-mark::after{content:'';position:absolute;inset:5px;border:1px solid var(--acc);border-radius:2px;opacity:.5}
.logo-cross{width:14px;height:14px;
  background:linear-gradient(var(--acc),var(--acc)) 50% 0/2px 100%,
             linear-gradient(var(--acc),var(--acc)) 0 50%/100% 2px;background-color:transparent}
@keyframes scanY{0%,100%{transform:translateY(0);opacity:1}50%{transform:translateY(26px);opacity:.4}}
.logo-cross{animation:scanY 3s ease-in-out infinite}
.logo-text h1{font-family:var(--mono);font-size:18px;letter-spacing:.18em;color:var(--acc);text-shadow:var(--glow)}
.logo-text p{font-size:10px;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-top:2px}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.track-pill{font-family:var(--mono);font-size:9px;letter-spacing:.12em;border:1px solid var(--brd);
  border-radius:3px;padding:3px 9px;color:var(--dim)}
.track-pill span{color:var(--acc)}
.eng-dot{width:7px;height:7px;border-radius:50%;background:var(--grn);
  box-shadow:0 0 10px rgba(0,217,126,.5);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ── top tabs (mode selector) ── */
.mode-tabs{display:flex;gap:0;border:1px solid var(--brd);border-radius:6px;overflow:hidden;margin-bottom:22px}
.mode-tab{flex:1;padding:10px 8px;background:transparent;border:none;border-right:1px solid var(--brd);
  color:var(--txtlo);font-family:var(--mono);font-size:10px;letter-spacing:.1em;cursor:pointer;
  text-transform:uppercase;transition:background .15s,color .15s;text-align:center}
.mode-tab:last-child{border-right:none}
.mode-tab:hover{background:rgba(0,229,255,.05);color:var(--txt)}
.mode-tab.active{background:rgba(0,229,255,.12);color:var(--acc);font-weight:600}
.mode-pane{display:none}
.mode-pane.active{display:block}

/* ── pipeline diagram ── */
.pipeline-diagram{background:var(--surf);border:1px solid var(--brd);border-radius:6px;
  padding:18px 20px;margin-bottom:22px;overflow-x:auto}
.pd-title{font-family:var(--mono);font-size:9px;letter-spacing:.18em;color:var(--dim);
  text-transform:uppercase;margin-bottom:14px}
.pd-flow{display:flex;align-items:center;gap:0;min-width:700px}
.pd-stage{display:flex;flex-direction:column;align-items:center;gap:5px;flex:1}
.pd-box{background:var(--panel);border:1px solid var(--brd);border-radius:5px;
  padding:9px 10px;text-align:center;width:100%;transition:border-color .2s,box-shadow .2s}
.pd-box.active{border-color:var(--acc);box-shadow:var(--glow)}
.pd-box.done{border-color:rgba(0,217,126,.4);background:rgba(0,217,126,.05)}
.pd-box.fail{border-color:rgba(255,68,68,.4);background:rgba(255,68,68,.05)}
.pd-box.warn{border-color:rgba(255,184,0,.35);background:rgba(255,184,0,.05)}
.pd-label{font-family:var(--mono);font-size:9px;letter-spacing:.1em;color:var(--dim);
  text-transform:uppercase;margin-bottom:3px}
.pd-val{font-family:var(--mono);font-size:17px;font-weight:700;color:var(--acc)}
.pd-arrow{font-size:18px;color:var(--brd);padding:0 4px;flex-shrink:0;margin-top:-14px}
.pd-sub{font-size:9px;color:var(--txtlo);margin-top:2px;letter-spacing:.04em}

/* ── two-col layout ── */
.main-grid{display:grid;grid-template-columns:360px 1fr;gap:20px;align-items:start}
@media(max-width:960px){.main-grid{grid-template-columns:1fr}}

/* ── panel ── */
.panel{background:var(--panel);border:1px solid var(--brd);border-radius:6px;padding:18px;margin-bottom:16px}
.ptitle{font-family:var(--mono);font-size:9px;letter-spacing:.18em;color:var(--dim);
  text-transform:uppercase;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--brd);
  display:flex;align-items:center;justify-content:space-between}
.ptag{font-size:8px;letter-spacing:.08em;padding:2px 6px;border-radius:2px;
  background:rgba(0,229,255,.1);color:var(--acc);border:1px solid rgba(0,229,255,.2)}

/* ── preset tabs ── */
.tab-row{display:flex;gap:0;border:1px solid var(--brd);border-radius:4px;overflow:hidden;margin-bottom:12px}
.tab-btn{flex:1;padding:7px 2px;background:transparent;border:none;border-right:1px solid var(--brd);
  color:var(--txtlo);font-family:var(--mono);font-size:8px;letter-spacing:.06em;cursor:pointer;
  text-transform:uppercase;transition:background .15s,color .15s;line-height:1.4;text-align:center}
.tab-btn:last-child{border-right:none}
.tab-btn:hover{background:rgba(0,229,255,.06);color:var(--txt)}
.tab-btn.active{background:rgba(0,229,255,.12);color:var(--acc)}
.tab-icon{display:block;font-size:13px;margin-bottom:1px}

/* ── form ── */
.field{margin-bottom:12px}
.field label{display:block;font-size:9px;letter-spacing:.12em;color:var(--dim);
  margin-bottom:4px;text-transform:uppercase;display:flex;justify-content:space-between;align-items:center}
.field textarea,.field select,.field input[type=text]{
  width:100%;background:var(--surf);border:1px solid var(--brd);border-radius:4px;
  color:var(--txt);font-family:var(--mono);font-size:11px;padding:7px 9px;outline:none;
  transition:border-color .15s;resize:vertical}
.field textarea:focus,.field select:focus,.field input:focus{border-color:var(--acc)}
.field textarea{min-height:90px;line-height:1.6}
.field-hint{font-size:9px;color:var(--txtlo);margin-top:3px}
.char-count{font-size:9px;color:var(--dim);font-family:var(--mono)}

/* ── corpus builder ── */
.corpus-builder{display:flex;flex-direction:column;gap:7px}
.corpus-item{display:flex;gap:6px;align-items:flex-start}
.corpus-item textarea{flex:1;min-height:60px;font-size:10px;padding:6px 8px}
.corpus-del{background:none;border:1px solid var(--brd);color:var(--dim);border-radius:3px;
  cursor:pointer;padding:4px 7px;font-size:11px;flex-shrink:0;transition:border-color .15s,color .15s}
.corpus-del:hover{border-color:var(--red);color:var(--red)}
.corpus-add{width:100%;padding:7px;background:transparent;border:1px dashed var(--brd);
  border-radius:4px;color:var(--dim);font-family:var(--mono);font-size:9px;letter-spacing:.1em;
  cursor:pointer;text-transform:uppercase;transition:border-color .15s,color .15s;margin-top:4px}
.corpus-add:hover{border-color:var(--acc);color:var(--acc)}

/* ── run button ── */
#run-btn{width:100%;padding:12px;background:transparent;border:2px solid var(--acc);
  border-radius:4px;color:var(--acc);font-family:var(--mono);font-size:12px;
  letter-spacing:.18em;cursor:pointer;text-transform:uppercase;box-shadow:var(--glow);
  transition:background .2s,color .2s,box-shadow .2s;margin-top:4px}
#run-btn:hover:not(:disabled){background:var(--acc);color:var(--bg);box-shadow:0 0 24px rgba(0,229,255,.6)}
#run-btn:disabled{opacity:.3;cursor:not-allowed}
.error-msg{background:rgba(255,68,68,.1);border:1px solid rgba(255,68,68,.3);border-radius:4px;
  padding:9px 12px;font-family:var(--mono);font-size:10px;color:#ff8a8a;margin-top:9px;display:none}

/* ── loading ── */
#loading{display:none;text-align:center;padding:50px 0}
.spinner{width:28px;height:28px;border:2px solid var(--brd);border-top-color:var(--acc);
  border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}
#loading p{font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:.1em}

/* ── results ── */
#result-panel{display:none}

/* ── verdict ring ── */
.verdict{display:flex;align-items:center;gap:22px;padding:18px;
  background:var(--surf);border:1px solid var(--brd);border-radius:6px;margin-bottom:16px}
.ring-wrap{position:relative;width:96px;height:96px;flex-shrink:0}
.ring-wrap svg{width:96px;height:96px}
.ring-center{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center}
.ring-num{font-family:var(--mono);font-size:24px;font-weight:700;line-height:1}
.ring-unit{font-size:8px;color:var(--dim);letter-spacing:.1em;margin-top:2px;text-transform:uppercase}
.verdict-meta{flex:1}
.v-pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:3px;
  font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px}
.v-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.v-desc{font-size:13px;color:var(--txtlo);line-height:1.5;margin-bottom:8px}
.v-conf{font-family:var(--mono);font-size:10px;color:var(--dim)}
.v-conf span{color:var(--txt)}

/* ── gate ── */
.gate{display:flex;align-items:center;gap:16px;padding:14px 16px;border-radius:6px;
  border:1px solid;margin-bottom:16px}
.gate-icon{font-size:22px;flex-shrink:0;width:36px;text-align:center}
.gate-body .gv{font-family:var(--mono);font-size:12px;font-weight:700;letter-spacing:.1em;margin-bottom:3px}
.gate-body .gd{font-size:12px;color:var(--txtlo);line-height:1.5}
.gate.pass{background:rgba(0,217,126,.07);border-color:rgba(0,217,126,.25)}
.gate.pass .gv{color:var(--grn)}
.gate.mixed{background:rgba(255,184,0,.07);border-color:rgba(255,184,0,.25)}
.gate.mixed .gv{color:var(--amb)}
.gate.fail{background:rgba(255,68,68,.07);border-color:rgba(255,68,68,.25)}
.gate.fail .gv{color:var(--red)}

/* ── 4-metric breakdown ── */
.breakdown{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}
.mc{background:var(--surf);border:1px solid var(--brd);border-radius:6px;padding:12px}
.mc-top{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:7px}
.mc-lbl{font-size:9px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase}
.mc-num{font-family:var(--mono);font-size:20px;font-weight:700;line-height:1}
.mc-bar-o{height:3px;background:var(--brd);border-radius:2px;overflow:hidden;margin-bottom:6px}
.mc-bar-i{height:100%;border-radius:2px;transition:width .7s ease}
.mc-note{font-size:10px;color:var(--txtlo);line-height:1.4}
.mc-inv{font-size:8px;color:var(--dim);letter-spacing:.04em;margin-top:2px}

/* ── feature family breakdown ── */
.fam-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}
.fam{background:var(--surf);border:1px solid var(--brd);border-radius:5px;padding:10px}
.fam-title{font-family:var(--mono);font-size:8px;letter-spacing:.1em;color:var(--acc);
  text-transform:uppercase;margin-bottom:8px;border-bottom:1px solid var(--brd);padding-bottom:5px}
.fam-row{display:flex;justify-content:space-between;align-items:center;padding:2px 0}
.fam-key{font-size:9px;color:var(--txtlo)}
.fam-val{font-family:var(--mono);font-size:9px;color:var(--txt)}

/* ── flags ── */
.flags{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}
.flag{display:flex;gap:8px;align-items:flex-start;padding:8px 11px;border-radius:4px;
  font-size:11px;line-height:1.5;border:1px solid}
.flag.good{background:rgba(0,217,126,.06);border-color:rgba(0,217,126,.18);color:#70e8a2}
.flag.bad{background:rgba(255,79,0,.06);border-color:rgba(255,79,0,.18);color:#ff9060}
.flag.warn{background:rgba(255,184,0,.06);border-color:rgba(255,184,0,.18);color:#ffd060}
.flag.info{background:rgba(0,229,255,.04);border-color:rgba(0,229,255,.14);color:#70ccdd}
.flag-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;margin-top:4px}
.flag.good .flag-dot{background:var(--grn)}
.flag.bad .flag-dot{background:var(--acc2)}
.flag.warn .flag-dot{background:var(--amb)}
.flag.info .flag-dot{background:var(--acc)}

/* ── suggestions ── */
.suggs{display:flex;flex-direction:column;gap:8px}
.sugg{padding:11px 13px;background:rgba(0,229,255,.03);
  border:1px solid rgba(0,229,255,.12);border-radius:4px}
.sugg strong{display:block;font-family:var(--mono);font-size:9px;
  letter-spacing:.1em;color:var(--acc);text-transform:uppercase;margin-bottom:3px}
.sugg p{font-size:11px;color:var(--txtlo);line-height:1.6}

/* ── session history ── */
.hist-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px}
.hcard{background:var(--surf);border:1px solid var(--brd);border-radius:5px;padding:10px;cursor:pointer;transition:border-color .15s}
.hcard:hover{border-color:var(--acc)}
.hc-ts{font-family:var(--mono);font-size:8px;color:var(--txtlo);letter-spacing:.06em;margin-bottom:5px}
.hc-sc{font-family:var(--mono);font-size:19px;font-weight:700;line-height:1;margin-bottom:3px}
.hc-lb{font-size:9px;color:var(--dim);letter-spacing:.06em}
.hc-bar{height:2px;background:var(--brd);border-radius:1px;overflow:hidden;margin-top:7px}
.hc-bi{height:100%;border-radius:1px}

/* ── batch mode ── */
.batch-row{display:flex;gap:8px;align-items:flex-start;padding:10px;
  background:var(--surf);border:1px solid var(--brd);border-radius:5px;margin-bottom:8px}
.br-idx{font-family:var(--mono);font-size:10px;color:var(--dim);width:22px;flex-shrink:0;padding-top:2px}
.br-fields{flex:1;display:flex;flex-direction:column;gap:6px}
.br-score{font-family:var(--mono);font-size:18px;font-weight:700;text-align:center;
  width:52px;flex-shrink:0;padding-top:4px}
.br-label{font-size:9px;text-align:center;letter-spacing:.06em;color:var(--dim);margin-top:2px}
.batch-add{width:100%;padding:8px;background:transparent;border:1px dashed var(--brd);
  border-radius:4px;color:var(--dim);font-family:var(--mono);font-size:9px;letter-spacing:.1em;
  cursor:pointer;text-transform:uppercase;transition:border-color .15s,color .15s;margin-bottom:10px}
.batch-add:hover{border-color:var(--acc);color:var(--acc)}
#batch-run-btn{width:100%;padding:11px;background:transparent;border:2px solid var(--acc);
  border-radius:4px;color:var(--acc);font-family:var(--mono);font-size:11px;
  letter-spacing:.18em;cursor:pointer;text-transform:uppercase;box-shadow:var(--glow);
  transition:background .2s,color .2s}
#batch-run-btn:hover:not(:disabled){background:var(--acc);color:var(--bg)}
#batch-run-btn:disabled{opacity:.3;cursor:not-allowed}

/* ── section label ── */
.sl{font-family:var(--mono);font-size:9px;letter-spacing:.2em;color:var(--dim);
  text-transform:uppercase;margin-bottom:9px;display:flex;align-items:center;gap:7px}
.sl::after{content:'';flex:1;height:1px;background:var(--brd)}
</style>
</head>
<body>
<div class="shell">

<!-- HEADER -->
<header>
  <div class="logo-mark"><div class="logo-cross"></div></div>
  <div class="logo-text">
    <h1>GRANT TRUST SYSTEM</h1>
    <p>End-to-End Intent Detection Pipeline</p>
  </div>
  <div class="hdr-right">
    <div class="track-pill">Track <span>2</span> + <span>3</span> · Apart Research 2026</div>
    <div class="eng-dot" title="Engine ready"></div>
  </div>
</header>

<!-- PIPELINE FLOW DIAGRAM -->
<div class="pipeline-diagram">
  <div class="pd-title">// Pipeline flow</div>
  <div class="pd-flow">
    <div class="pd-stage">
      <div class="pd-box" id="pd-input">
        <div class="pd-label">Input</div>
        <div class="pd-val" id="pd-input-val">—</div>
        <div class="pd-sub">application + corpus</div>
      </div>
    </div>
    <div class="pd-arrow">→</div>
    <div class="pd-stage">
      <div class="pd-box" id="pd-features">
        <div class="pd-label">Features</div>
        <div class="pd-val" id="pd-feat-val">42</div>
        <div class="pd-sub">stylometric</div>
      </div>
    </div>
    <div class="pd-arrow">→</div>
    <div class="pd-stage">
      <div class="pd-box" id="pd-auth">
        <div class="pd-label">Authorship</div>
        <div class="pd-val" id="pd-auth-val">—</div>
        <div class="pd-sub">vs corpus</div>
      </div>
    </div>
    <div class="pd-arrow">→</div>
    <div class="pd-stage">
      <div class="pd-box" id="pd-tmpl">
        <div class="pd-label">Template</div>
        <div class="pd-val" id="pd-tmpl-val">—</div>
        <div class="pd-sub">DBSCAN dist</div>
      </div>
    </div>
    <div class="pd-arrow">→</div>
    <div class="pd-stage">
      <div class="pd-box" id="pd-spray">
        <div class="pd-label">Spray</div>
        <div class="pd-val" id="pd-spray-val">—</div>
        <div class="pd-sub">uniformity</div>
      </div>
    </div>
    <div class="pd-arrow">→</div>
    <div class="pd-stage">
      <div class="pd-box" id="pd-spec">
        <div class="pd-label">Specificity</div>
        <div class="pd-val" id="pd-spec-val">—</div>
        <div class="pd-sub">engagement</div>
      </div>
    </div>
    <div class="pd-arrow">→</div>
    <div class="pd-stage">
      <div class="pd-box" id="pd-intent">
        <div class="pd-label">Intent</div>
        <div class="pd-val" id="pd-intent-val">—</div>
        <div class="pd-sub">composite</div>
      </div>
    </div>
    <div class="pd-arrow">→</div>
    <div class="pd-stage">
      <div class="pd-box" id="pd-gate">
        <div class="pd-label">Gate</div>
        <div class="pd-val" id="pd-gate-val">—</div>
        <div class="pd-sub">control</div>
      </div>
    </div>
  </div>
</div>

<!-- MODE TABS -->
<div class="mode-tabs">
  <button class="mode-tab active" onclick="switchMode('single')"  id="mtab-single">Single analysis</button>
  <button class="mode-tab"        onclick="switchMode('batch')"   id="mtab-batch">Batch analysis</button>
  <button class="mode-tab"        onclick="switchMode('history')" id="mtab-history">Session history</button>
</div>

<!-- ─── SINGLE MODE ─── -->
<div class="mode-pane active" id="pane-single">
<div class="main-grid">

  <!-- LEFT -->
  <aside>
    <div class="panel">
      <div class="ptitle">// Demo presets <span class="ptag">hackathon cases</span></div>
      <div class="tab-row">
        <button class="tab-btn active" onclick="loadPreset('a')"     id="tab-a"><span class="tab-icon">◈</span>Genuine</button>
        <button class="tab-btn"        onclick="loadPreset('b')"     id="tab-b"><span class="tab-icon">⊗</span>Spray</button>
        <button class="tab-btn"        onclick="loadPreset('c')"     id="tab-c"><span class="tab-icon">◎</span>Wrong fit</button>
        <button class="tab-btn"        onclick="loadPreset('b2')"    id="tab-b2"><span class="tab-icon">↻</span>Improved</button>
        <button class="tab-btn"        onclick="loadPreset('blank')" id="tab-blank"><span class="tab-icon">+</span>Custom</button>
      </div>
      <div id="preset-desc" style="font-size:10px;color:var(--txtlo);margin-bottom:12px;line-height:1.5"></div>

      <div class="field">
        <label>Funder profile</label>
        <select id="funder-select">
          <option value="deep_science_ventures">Deep Science Ventures</option>
          <option value="civic_tech_fund">Civic Tech Fund</option>
          <option value="ai_safety_institute">AI Safety Institute</option>
        </select>
      </div>
    </div>

    <div class="panel">
      <div class="ptitle">// Application text <span class="ptag">agent output</span></div>
      <div class="field">
        <label>
          <span>Grant application</span>
          <span class="char-count" id="app-chars">0 words</span>
        </label>
        <textarea id="app-text" rows="9"
          placeholder="Paste the grant application to analyse…"
          oninput="updateCounts()"></textarea>
      </div>
    </div>

    <div class="panel">
      <div class="ptitle">// Principal corpus <span class="ptag">costly signal</span></div>
      <div id="corpus-builder" class="corpus-builder"></div>
      <button class="corpus-add" onclick="addCorpusItem()">+ add writing sample</button>
      <div class="field-hint" style="margin-top:6px">Prior writing ground-truths the principal's voice. Leave empty to test corpus-free detection.</div>
    </div>

    <div class="panel">
      <div class="ptitle">// Other applications <span class="ptag">spray detector</span></div>
      <div class="field">
        <textarea id="other-apps" rows="3"
          placeholder="Paste another application from this person to enable cross-application uniformity detection…"></textarea>
        <div class="field-hint">Enables the COPY_PASTE_RISK flag. Leave empty for intrinsic-only spray detection.</div>
      </div>
    </div>

    <button id="run-btn" onclick="runAnalysis()">▶ RUN INTENT ANALYSIS</button>
    <div class="error-msg" id="error-msg"></div>
  </aside>

  <!-- RIGHT -->
  <div>
    <div id="loading">
      <div class="spinner"></div>
      <p>Running stylometric pipeline…</p>
    </div>

    <div id="result-panel">

      <!-- Verdict ring -->
      <div class="verdict">
        <div class="ring-wrap">
          <svg viewBox="0 0 96 96">
            <circle cx="48" cy="48" r="40" fill="none" stroke="#1a2744" stroke-width="6"/>
            <circle id="ring-arc" cx="48" cy="48" r="40" fill="none" stroke-width="6"
              stroke-linecap="round" stroke-dasharray="251.3" stroke-dashoffset="251.3"
              transform="rotate(-90 48 48)" style="transition:stroke-dashoffset .9s ease,stroke .4s"/>
          </svg>
          <div class="ring-center">
            <div class="ring-num" id="ring-num">—</div>
            <div class="ring-unit">intent</div>
          </div>
        </div>
        <div class="verdict-meta">
          <div class="v-pill" id="v-pill">
            <div class="v-dot" id="v-dot"></div>
            <span id="v-label">—</span>
          </div>
          <div class="v-desc" id="v-desc">Run an analysis to see results.</div>
          <div class="v-conf">Confidence: <span id="v-conf">—</span></div>
        </div>
      </div>

      <!-- Control gate -->
      <div class="gate" id="gate-box">
        <div class="gate-icon" id="gate-icon">—</div>
        <div class="gate-body">
          <div class="gv" id="gate-verdict">—</div>
          <div class="gd" id="gate-detail">—</div>
        </div>
      </div>

      <!-- 4-metric breakdown -->
      <div class="sl">Component scores</div>
      <div class="breakdown">
        <div class="mc">
          <div class="mc-top"><div class="mc-lbl">Authorship</div><div class="mc-num" id="m-auth" style="color:var(--acc)">—</div></div>
          <div class="mc-bar-o"><div class="mc-bar-i" id="b-auth" style="background:var(--acc);width:0%"></div></div>
          <div class="mc-note">Stylometric match to principal's prior writing</div>
        </div>
        <div class="mc">
          <div class="mc-top"><div class="mc-lbl">Template distance</div><div class="mc-num" id="m-tmpl" style="color:var(--acc)">—</div></div>
          <div class="mc-bar-o"><div class="mc-bar-i" id="b-tmpl" style="background:var(--acc);width:0%"></div></div>
          <div class="mc-note">Distance from LLM baseline clusters</div>
        </div>
        <div class="mc">
          <div class="mc-top"><div class="mc-lbl">Spray score</div><div class="mc-num" id="m-spray" style="color:var(--acc2)">—</div></div>
          <div class="mc-bar-o"><div class="mc-bar-i" id="b-spray" style="background:var(--acc2);width:0%"></div></div>
          <div class="mc-note">Volume side-task signal</div>
          <div class="mc-inv">inverted in final formula</div>
        </div>
        <div class="mc">
          <div class="mc-top"><div class="mc-lbl">Specificity</div><div class="mc-num" id="m-spec" style="color:var(--grn)">—</div></div>
          <div class="mc-bar-o"><div class="mc-bar-i" id="b-spec" style="background:var(--grn);width:0%"></div></div>
          <div class="mc-note">Concrete evidence of genuine engagement</div>
        </div>
      </div>

      <!-- Feature family breakdown -->
      <div class="sl">Feature families <span style="color:var(--txtlo);font-family:var(--mono);font-size:9px;margin-left:4px">(42 features)</span></div>
      <div class="fam-grid" id="fam-grid">
        <div class="fam">
          <div class="fam-title">Lexical (10)</div>
          <div id="fam-lexical"></div>
        </div>
        <div class="fam">
          <div class="fam-title">Syntactic (12)</div>
          <div id="fam-syntactic"></div>
        </div>
        <div class="fam">
          <div class="fam-title">Surface (8)</div>
          <div id="fam-surface"></div>
        </div>
        <div class="fam">
          <div class="fam-title">Semantic (12)</div>
          <div id="fam-semantic"></div>
        </div>
      </div>

      <!-- Flags -->
      <div class="sl">Detection flags</div>
      <div class="flags" id="flags-list"></div>

      <!-- Suggestions -->
      <div class="sl">Improvement pathways → component 3</div>
      <div class="suggs" id="suggs-list"></div>

    </div><!-- /result-panel -->
  </div>
</div>
</div><!-- /pane-single -->

<!-- ─── BATCH MODE ─── -->
<div class="mode-pane" id="pane-batch">
  <div class="panel">
    <div class="ptitle">// Batch analysis <span class="ptag">run multiple applications</span></div>
    <div id="batch-rows"></div>
    <button class="batch-add" onclick="addBatchRow()">+ add application</button>
    <div class="field" style="margin-top:12px">
      <label>Funder profile (applies to all)</label>
      <select id="batch-funder">
        <option value="deep_science_ventures">Deep Science Ventures</option>
        <option value="civic_tech_fund">Civic Tech Fund</option>
        <option value="ai_safety_institute">AI Safety Institute</option>
      </select>
    </div>
    <button id="batch-run-btn" onclick="runBatch()">▶ RUN BATCH ANALYSIS</button>
    <div class="error-msg" id="batch-error"></div>
  </div>
</div>

<!-- ─── HISTORY MODE ─── -->
<div class="mode-pane" id="pane-history">
  <div class="panel">
    <div class="ptitle">// Session history <span class="ptag" id="hist-count">0 runs</span></div>
    <div id="hist-grid-outer">
      <div class="hist-grid" id="hist-grid"></div>
    </div>
    <div id="hist-empty" style="text-align:center;padding:40px 0;font-family:var(--mono);font-size:11px;color:var(--dim)">
      No analyses run yet in this session.
    </div>
  </div>
</div>

</div><!-- /shell -->

<script>
const API = window.location.origin;

const LABEL_COLOR = {
  spray_and_pray:'#ff4444', likely_automated:'#ff6620',
  mixed:'#ffb800', likely_genuine:'#00e5ff', genuine:'#00d97e',
};
const LABEL_DESC = {
  spray_and_pray: 'Agent side-task detected with high confidence. Volume-maximization behavior. No meaningful principal oversight.',
  likely_automated: 'Likely agent-generated with minimal human curation. Low authorship match, LLM template proximity.',
  mixed: 'Ambiguous signals — monitor uncertain. Interrogation protocol should fire before gate decision.',
  likely_genuine: 'Principal engagement evident. Stylistically consistent with researcher\'s own writing.',
  genuine: 'Strong human-in-the-loop signal. Application reflects the principal\'s own thinking, augmented by AI.',
};

// ── Feature family labels (in vector order) ──────────────────────────────
const FAM_LABELS = {
  lexical:   ['TTR','hapax rate','avg word len','std word len','Yule K','Brunet W','rare ratio','freq entropy','long word ratio','contraction rate'],
  syntactic: ['avg sent len','std sent len','sent entropy','comma rate','dash rate','paren rate','semicolon rate','question rate','exclaim rate','n paragraphs','avg para len','sub-clause rate'],
  surface:   ['Flesch ease','FK grade','Gunning fog','log length','avg char/word','digit ratio','upper ratio','sent/word ratio'],
  semantic:  ['LLM cliché density','specificity markers','fp ratio','hedge rate','passive rate','NE density','numeric density','tech density','temporal density','evidence rate','URL rate','citation rate'],
};

// ── Presets ──────────────────────────────────────────────────────────────
let PRESETS = {};
const PRESET_DESC = {
  a:     'Genuine researcher (A) · AI-assisted, strong corpus, own voice. Expected: likely_genuine.',
  b:     'Spray-and-pray (B) · No corpus, pure LLM template. Expected: likely_automated.',
  c:     'Wrong funder (C) · Real researcher, poor mission alignment. Expected: likely_genuine → fails fit gate.',
  b2:    'Post-intervention (B) · After hackathon + shipped project. Expected: mixed (feedback loop).',
  blank: 'Custom · Clear all fields and enter your own text.',
};

async function loadPresets() {
  try {
    const r = await fetch(`${API}/api/demo_data`);
    PRESETS = await r.json();
    loadPreset('a');
    initBatchRows();
  } catch(e) { console.error('Failed to load presets', e); }
}

// ── Corpus builder ────────────────────────────────────────────────────────
let corpusItems = [];
function renderCorpusBuilder() {
  const cb = document.getElementById('corpus-builder');
  cb.innerHTML = '';
  corpusItems.forEach(function(text, i) {
    const row = document.createElement('div');
    row.className = 'corpus-item';
    const ta = document.createElement('textarea');
    ta.className = 'field';
    ta.rows = 3;
    ta.value = text;
    ta.placeholder = 'Blog post, README, hackathon write-up…';
    ta.oninput = function() { corpusItems[i] = ta.value; };
    const btn = document.createElement('button');
    btn.className = 'corpus-del';
    btn.textContent = '×';
    btn.onclick = function() { corpusItems.splice(i,1); renderCorpusBuilder(); };
    row.appendChild(ta);
    row.appendChild(btn);
    cb.appendChild(row);
  });
}
function addCorpusItem() {
  corpusItems.push('');
  renderCorpusBuilder();
  const tas = document.querySelectorAll('#corpus-builder textarea');
  if (tas.length) tas[tas.length-1].focus();
}

function loadPreset(key) {
  document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById('tab-'+key).classList.add('active');
  document.getElementById('preset-desc').textContent = PRESET_DESC[key] || '';
  if (key === 'blank') {
    document.getElementById('app-text').value = '';
    corpusItems = [];
    renderCorpusBuilder();
    document.getElementById('other-apps').value = '';
    document.getElementById('funder-select').value = 'deep_science_ventures';
    updateCounts();
    return;
  }
  const appMap = {a:'applicant_a', b:'applicant_b_spray', c:'applicant_c', b2:'applicant_b_improved'};
  const corpMap = {a:'applicant_a', b:'applicant_b', c:'applicant_c', b2:'applicant_b'};
  const otherMap = {b:'applicant_b_v2'};
  document.getElementById('app-text').value = (PRESETS.applications||{})[appMap[key]]||'';
  corpusItems = [...((PRESETS.corpus||{})[corpMap[key]]||[])];
  renderCorpusBuilder();
  document.getElementById('other-apps').value = otherMap[key] ? ((PRESETS.applications||{})[otherMap[key]]||'') : '';
  document.getElementById('funder-select').value = 'deep_science_ventures';
  updateCounts();
  resetPipeline();
  document.getElementById('result-panel').style.display = 'none';
}

function updateCounts() {
  const words = document.getElementById('app-text').value.trim().split(/\s+/).filter(Boolean).length;
  document.getElementById('app-chars').textContent = words + ' words';
  // Update pipeline input stage
  document.getElementById('pd-input-val').textContent = words > 0 ? words+'w' : '—';
}

// ── Mode switching ────────────────────────────────────────────────────────
function switchMode(mode) {
  document.querySelectorAll('.mode-tab').forEach(function(t){t.classList.remove('active');});
  document.querySelectorAll('.mode-pane').forEach(function(p){p.classList.remove('active');});
  document.getElementById('mtab-'+mode).classList.add('active');
  document.getElementById('pane-'+mode).classList.add('active');
  if (mode === 'history') renderHistory();
}

// ── Pipeline diagram helpers ──────────────────────────────────────────────
function resetPipeline() {
  ['pd-input','pd-features','pd-auth','pd-tmpl','pd-spray','pd-spec','pd-intent','pd-gate'].forEach(function(id){
    const el = document.getElementById(id);
    if (el) { el.className='pd-box'; }
  });
  ['pd-auth-val','pd-tmpl-val','pd-spray-val','pd-spec-val','pd-intent-val','pd-gate-val'].forEach(function(id){
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  });
}

function updatePipeline(d) {
  const pct = function(v){return Math.round(v*100)+'%';};
  document.getElementById('pd-features').classList.add('done');
  document.getElementById('pd-feat-val').textContent = '42';

  document.getElementById('pd-auth-val').textContent = pct(d.authorship_score);
  document.getElementById('pd-auth').classList.add(d.authorship_score > 0.6 ? 'done' : 'warn');

  document.getElementById('pd-tmpl-val').textContent = pct(d.template_score);
  document.getElementById('pd-tmpl').classList.add(d.template_score > 0.5 ? 'done' : 'warn');

  document.getElementById('pd-spray-val').textContent = pct(d.spray_score);
  document.getElementById('pd-spray').classList.add(d.spray_score < 0.4 ? 'done' : d.spray_score > 0.65 ? 'fail' : 'warn');

  document.getElementById('pd-spec-val').textContent = pct(d.specificity_score);
  document.getElementById('pd-spec').classList.add(d.specificity_score > 0.4 ? 'done' : 'warn');

  document.getElementById('pd-intent-val').textContent = Math.round(d.intent_score*100);
  const intentClass = d.intent_score >= 0.65 ? 'done' : d.intent_score >= 0.45 ? 'warn' : 'fail';
  document.getElementById('pd-intent').classList.add(intentClass);

  const gateClass = d.intent_score >= 0.65 ? 'done' : d.intent_score >= 0.45 ? 'warn' : 'fail';
  const gateText  = d.intent_score >= 0.65 ? 'PASS' : d.intent_score >= 0.45 ? 'MIXED' : 'FAIL';
  document.getElementById('pd-gate-val').textContent = gateText;
  document.getElementById('pd-gate').classList.add(gateClass);
}

// ── Single analysis ───────────────────────────────────────────────────────
const sessionHistory = [];

async function runAnalysis() {
  const appText = document.getElementById('app-text').value.trim();
  const errEl   = document.getElementById('error-msg');
  errEl.style.display = 'none';
  if (!appText) { showError('Please enter application text.', 'error-msg'); return; }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('result-panel').style.display = 'none';
  document.getElementById('run-btn').disabled = true;
  resetPipeline();
  document.getElementById('pd-input').classList.add('active');

  const corpus = corpusItems.filter(function(t){return t.trim().length > 20;});
  const others = document.getElementById('other-apps').value.trim();
  const funder = document.getElementById('funder-select').value;

  try {
    const r = await fetch(`${API}/api/analyze`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        application_text: appText,
        corpus: corpus,
        other_applications: others ? [others] : [],
        funder_id: funder,
      })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderResult(d);
    updatePipeline(d);
    sessionHistory.unshift({...d, ts: new Date().toLocaleTimeString(), snippet: appText.slice(0,50)+'…', funder});
    if (sessionHistory.length > 10) sessionHistory.pop();
  } catch(e) {
    showError(e.message, 'error-msg');
    resetPipeline();
  } finally {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
    document.getElementById('pd-input').classList.remove('active');
    document.getElementById('pd-input').classList.add('done');
  }
}

function showError(msg, id) {
  const el = document.getElementById(id);
  el.textContent = 'Error: '+msg;
  el.style.display = 'block';
}

function renderResult(d) {
  const score = d.intent_score;
  const col   = LABEL_COLOR[d.label] || 'var(--acc)';
  const circ  = 2 * Math.PI * 40;
  document.getElementById('ring-arc').style.strokeDashoffset = circ * (1 - score);
  document.getElementById('ring-arc').style.stroke = col;
  document.getElementById('ring-num').textContent = Math.round(score * 100);
  document.getElementById('ring-num').style.color = col;

  const pill = document.getElementById('v-pill');
  pill.style.background = col+'22'; pill.style.border = '1px solid '+col+'55'; pill.style.color = col;
  document.getElementById('v-dot').style.background = col;
  document.getElementById('v-label').textContent = d.label.replace(/_/g,' ');
  document.getElementById('v-desc').textContent = LABEL_DESC[d.label] || '';
  document.getElementById('v-conf').textContent = Math.round(d.confidence*100)+'%';

  // Gate
  const gate = document.getElementById('gate-box');
  gate.className = 'gate';
  if (score >= 0.65) {
    gate.classList.add('pass');
    document.getElementById('gate-icon').textContent = '✓';
    document.getElementById('gate-verdict').textContent = 'PASS — Control gate cleared';
    document.getElementById('gate-detail').textContent = 'Application surfaces to funder. Principal engagement meets threshold.';
  } else if (d.label === 'mixed') {
    gate.classList.add('mixed');
    document.getElementById('gate-icon').textContent = '?';
    document.getElementById('gate-verdict').textContent = 'MIXED — Interrogation protocol triggered';
    document.getElementById('gate-detail').textContent = 'Monitor is uncertain. Trusted reviewer asks the principal factual questions before the gate fires.';
  } else {
    gate.classList.add('fail');
    document.getElementById('gate-icon').textContent = '✕';
    document.getElementById('gate-verdict').textContent = 'FAIL — Improvement pathway generated';
    document.getElementById('gate-detail').textContent = 'Below threshold. Actionable next steps delivered below. Complete them and re-apply.';
  }

  // Component bars
  setM('auth',  d.authorship_score,  'var(--acc)');
  setM('tmpl',  d.template_score,    'var(--acc)');
  setM('spray', d.spray_score,       'var(--acc2)');
  setM('spec',  d.specificity_score, 'var(--grn)');

  // Feature families (from detail if present, else show placeholder rows)
  if (d.feature_detail) {
    const famDivergence = (d.feature_detail.authorship_detail || {}).family_divergence || {};
    renderFamilies(famDivergence, d);
  }

  // Flags
  const fl = document.getElementById('flags-list');
  fl.innerHTML = '';
  (d.flags || []).forEach(function(f) {
    const g = /MATCH|STRONG|HIGH_SPEC/.test(f), b = /SPRAY|TEMPLATE|COPY|GENERIC|DIVERGENCE/.test(f);
    fl.innerHTML += '<div class="flag '+(g?'good':b?'bad':'warn')+'"><div class="flag-dot"></div><div>'+f+'</div></div>';
  });
  if (!(d.flags||[]).length) fl.innerHTML = '<div class="flag info"><div class="flag-dot"></div><div>No specific flags fired.</div></div>';

  // Suggestions
  const sl = document.getElementById('suggs-list');
  sl.innerHTML = '';
  (d.suggestions || []).forEach(function(s) {
    const ci = s.indexOf(':');
    const k  = ci>=0 ? s.slice(0,ci) : '';
    const b  = ci>=0 ? s.slice(ci+1).trim() : s;
    sl.innerHTML += '<div class="sugg"><strong>'+k+'</strong><p>'+b+'</p></div>';
  });

  document.getElementById('result-panel').style.display = 'block';
}

function setM(key, val, col) {
  document.getElementById('m-'+key).textContent = Math.round(val*100)+'%';
  document.getElementById('b-'+key).style.width = (val*100).toFixed(1)+'%';
}

function renderFamilies(famDiv, d) {
  const families = ['lexical','syntactic','surface','semantic'];
  const divScores = {
    lexical:   d.authorship_score,
    syntactic: d.template_score,
    surface:   d.specificity_score,
    semantic:  1 - d.spray_score,
  };
  families.forEach(function(fam) {
    const el = document.getElementById('fam-'+fam);
    if (!el) return;
    const labels = FAM_LABELS[fam] || [];
    // Show divergence score + a couple of key labels
    const div = famDiv[fam];
    let html = '';
    if (div !== undefined) {
      const divPct = Math.round(div * 100);
      const divCol = div > 0.3 ? 'var(--acc2)' : div > 0.15 ? 'var(--amb)' : 'var(--grn)';
      html += '<div class="fam-row"><div class="fam-key">divergence</div><div class="fam-val" style="color:'+divCol+'">'+divPct+'%</div></div>';
    }
    // Show 3 representative feature names
    const show = labels.slice(0,3);
    show.forEach(function(lbl) {
      html += '<div class="fam-row"><div class="fam-key">'+lbl+'</div><div class="fam-val">·</div></div>';
    });
    if (labels.length > 3) html += '<div class="fam-row"><div class="fam-key" style="color:var(--dim)">+'+(labels.length-3)+' more</div></div>';
    el.innerHTML = html;
  });
}

// ── Batch mode ────────────────────────────────────────────────────────────
let batchRows = [];
function initBatchRows() {
  batchRows = [
    {text:(PRESETS.applications||{}).applicant_a||'', label:'Applicant A'},
    {text:(PRESETS.applications||{}).applicant_b_spray||'', label:'Applicant B'},
    {text:(PRESETS.applications||{}).applicant_c||'', label:'Applicant C'},
  ];
  renderBatchRows();
}
function addBatchRow() {
  batchRows.push({text:'', label:'Applicant '+(batchRows.length+1)});
  renderBatchRows();
}
function renderBatchRows() {
  const container = document.getElementById('batch-rows');
  container.innerHTML = '';
  batchRows.forEach(function(row, i) {
    const div = document.createElement('div');
    div.className = 'batch-row';
    div.innerHTML = '<div class="br-idx">'+(i+1)+'</div>'
      +'<div class="br-fields">'
      +'<input type="text" placeholder="Label (e.g. Applicant A)" value="'+row.label+'" '
      +'oninput="batchRows['+i+'].label=this.value" style="font-size:10px;padding:5px 8px;background:var(--surf);border:1px solid var(--brd);border-radius:3px;color:var(--txt);font-family:var(--mono);width:100%;outline:none"/>'
      +'<textarea rows="3" placeholder="Application text…" '
      +'oninput="batchRows['+i+'].text=this.value" style="font-size:10px">'
      +row.text+'</textarea>'
      +'</div>'
      +'<div>'
      +'<div class="br-score" id="bs-'+i+'" style="color:var(--dim)">—</div>'
      +'<div class="br-label" id="bl-'+i+'">—</div>'
      +'</div>';
    container.appendChild(div);
  });
}
async function runBatch() {
  const btn = document.getElementById('batch-run-btn');
  const errEl = document.getElementById('batch-error');
  errEl.style.display = 'none';
  btn.disabled = true;
  btn.textContent = '⟳ Running…';
  const funder = document.getElementById('batch-funder').value;
  const valid = batchRows.filter(function(r){return r.text.trim().length > 20;});
  if (!valid.length) { showError('Add at least one application.','batch-error'); btn.disabled=false; btn.textContent='▶ RUN BATCH ANALYSIS'; return; }
  try {
    const results = await Promise.all(batchRows.map(async function(row, i) {
      if (!row.text.trim()) {
        document.getElementById('bs-'+i).textContent = '—';
        document.getElementById('bl-'+i).textContent = '—';
        return null;
      }
      const r = await fetch(`${API}/api/analyze`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({application_text:row.text, corpus:[], funder_id:funder})
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      const col = LABEL_COLOR[d.label]||'var(--acc)';
      document.getElementById('bs-'+i).textContent = Math.round(d.intent_score*100);
      document.getElementById('bs-'+i).style.color = col;
      document.getElementById('bl-'+i).textContent = d.label.replace(/_/g,' ');
      document.getElementById('bl-'+i).style.color = col;
      sessionHistory.unshift({...d, ts:new Date().toLocaleTimeString(), snippet:row.label, funder});
      return d;
    }));
  } catch(e) {
    showError(e.message, 'batch-error');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ RUN BATCH ANALYSIS';
  }
}

// ── Session history ───────────────────────────────────────────────────────
function renderHistory() {
  const grid   = document.getElementById('hist-grid');
  const empty  = document.getElementById('hist-empty');
  const count  = document.getElementById('hist-count');
  count.textContent = sessionHistory.length + ' run' + (sessionHistory.length !== 1 ? 's' : '');
  if (!sessionHistory.length) {
    grid.style.display = 'none';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';
  grid.style.display = 'grid';
  grid.innerHTML = '';
  sessionHistory.forEach(function(h) {
    const col = LABEL_COLOR[h.label]||'var(--acc)';
    const pct = Math.round(h.intent_score*100);
    const c = document.createElement('div');
    c.className = 'hcard';
    c.innerHTML = '<div class="hc-ts">'+h.ts+'</div>'
      +'<div class="hc-sc" style="color:'+col+'">'+pct+'</div>'
      +'<div class="hc-lb" style="color:'+col+'">'+h.label.replace(/_/g,' ')+'</div>'
      +'<div style="font-size:9px;color:var(--txtlo);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+h.snippet+'</div>'
      +'<div class="hc-bar"><div class="hc-bi" style="width:'+pct+'%;background:'+col+'"></div></div>';
    grid.appendChild(c);
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────
loadPresets();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES  —  single + batch analyze + full feature detail
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(FRONTEND)

@app.route("/api/status")
def api_status():
    return jsonify({"ready": True})

@app.route("/api/demo_data")
def api_demo_data():
    return jsonify({"applications": DEMO_APPLICATIONS, "corpus": DEMO_CORPUS})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data     = request.get_json(force=True) or {}
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
            "feature_detail":    result.feature_detail,   # full detail for family breakdown
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Analyze multiple applications in one call."""
    data     = request.get_json(force=True) or {}
    items    = data.get("items", [])
    funder_id = data.get("funder_id", "deep_science_ventures")
    funder   = DEMO_FUNDERS.get(funder_id)

    if not items:
        return jsonify({"error": "items array is required"}), 400

    results = []
    for item in items:
        app_text = item.get("application_text", "").strip()
        if not app_text:
            results.append({"error": "empty text"})
            continue
        corpus = ApplicantCorpus(applicant_id="user",
                                 texts=item.get("corpus", []))
        try:
            r = ENGINE.analyze(
                application_text   = app_text,
                corpus             = corpus,
                other_applications = item.get("other_applications", []),
                funder_context     = funder,
            )
            results.append({
                "label":             r.label,
                "intent_score":      r.intent_score,
                "confidence":        r.confidence,
                "authorship_score":  r.authorship_score,
                "template_score":    r.template_score,
                "spray_score":       r.spray_score,
                "specificity_score": r.specificity_score,
                "flags":             r.flags,
            })
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify({"results": results})


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    host = os.environ.get("HOST", "0.0.0.0")
    print("=" * 60)
    print("GRANT TRUST SYSTEM — End-to-End Pipeline")
    print(f"  Engine : {ENGINE.__class__.__name__}")
    print(f"  Funders: {list(DEMO_FUNDERS.keys())}")
    print(f"  Serving: http://{host}:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False, threaded=True)
