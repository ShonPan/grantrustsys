"""
Grant Trust System — Intent Detection Engine
============================================
Hackathon spec: Caleb Strom + Shon Pan
Component 1 of 3: Intent detection engine (Caleb — primary)

Adapts the Xenarch stylometric pipeline from planetary anomaly detection
to grant application intent scoring. The core insight from the spec:

  "The pattern of AI usage is the signal, not the fact of AI usage."

Architecture mirrors Xenarch's multi-metric scoring system:
  Application text ──→ [Stylometric feature extraction]
                   ──→ [DBSCAN clustering vs. LLM baselines]  → Template score
                   ──→ [Compare vs. applicant prior corpus]   → Authorship score
                   ──→ [Cross-application uniformity check]   → Spray score

Output: Intent score on spectrum from "automated spray-and-pray"
        to "genuine human-in-the-loop curation"

Dependencies:
    pip install numpy scipy scikit-learn spacy textstat langdetect
    python -m spacy download en_core_web_sm
"""

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

if __name__ == "__main__":

    engine = IntentDetectionEngine()

    # ── Funder context ────────────────────────────────────────────────────
    funder = FunderContext(
        funder_id            = "deep_science_ventures",
        mission_keywords     = ["frontier science", "planetary", "technosignature",
                                "anomaly detection", "remote sensing", "ML for science"],
        focus_areas          = ["astrobiology", "planetary science", "AI/ML", "open science"],
        past_recipient_language = ["validated on real orbital data", "open-source pipeline",
                                   "reproducible methodology"]
    )

    # ── Applicant A: Genuine researcher (Caleb-style) ─────────────────────
    corpus_a = ApplicantCorpus(
        applicant_id = "applicant_a",
        texts = [
            # Simulates blog post / README in Caleb's voice
            """
            I've been working on Xenarch for about six months now, and version 2 finally
            cracked the training stability problem that was killing me in v1. The trick was
            combining gradient clipping at 1.0 with KL annealing over the first three epochs.
            I lost count of how many runs hit NaN at epoch 8 before I figured that out.
            The Apollo 11 test is a nice sanity check — if your model can't rank the lander
            as the top anomaly in LROC imagery, something is fundamentally wrong with your
            scoring. The contextual metric is doing most of the work for circular features.
            """,
            """
            The latent space geometry of the VAE matters more than I initially thought.
            With z=56 and proper KL regularisation, the model learns to cluster geological
            textures that share formation mechanisms — not just visual similarity. A fresh
            crater and a degraded crater end up in different parts of the space even though
            they look similar at first glance. I think this is why the false positive rate
            on boulder fields dropped so sharply between v1 and v2.
            """
        ]
    )

    application_a = """
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

    # ── Applicant B: Spray-and-pray ───────────────────────────────────────
    corpus_b = ApplicantCorpus(applicant_id="applicant_b", texts=[])  # no corpus

    application_b_v1 = """
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

    # Simulates sending the same template to a different funder
    application_b_v2 = """
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

    # ── Applicant C: Real person, wrong funder ────────────────────────────
    corpus_c = ApplicantCorpus(
        applicant_id = "applicant_c",
        texts = [
            """
            I spent most of last year building a distributed sensor network for urban
            air quality monitoring. The hard part wasn't the hardware — it was getting
            the calibration to hold up across different weather conditions. I ended up
            training a gradient-boosted model on the sensor drift patterns and got
            RMSE down to 2.3 µg/m³ across 47 deployment sites.
            """
        ]
    )

    application_c = """
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

    # ── Run the demo ───────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("GRANT TRUST SYSTEM — INTENT DETECTION ENGINE DEMO")
    print("="*70)

    demo_cases = [
        ("APPLICANT A: Genuine researcher, AI-augmented",
         application_a, corpus_a, None, funder),
        ("APPLICANT B: Spray-and-pray (first application)",
         application_b_v1, corpus_b, [application_b_v2], funder),
        ("APPLICANT C: Real person, wrong funder",
         application_c, corpus_c, None, funder),
    ]

    for title, app_text, corpus, other_apps, funder_ctx in demo_cases:
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print("─"*70)

        result = engine.analyze(
            application_text    = app_text,
            corpus              = corpus,
            other_applications  = other_apps,
            funder_context      = funder_ctx,
        )

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
        print(f"\n  Suggestions for component 3:")
        for s in result.suggestions:
            print(f"    → {s}")

    print(f"\n{'='*70}\n")
