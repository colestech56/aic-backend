"""
Content Screener for LLM-generated intervention messages.

Checks all generated content against the attribution framework rules
before delivery. Section 6.3 of the AIC spec.

Screening checks:
1. Forbidden keyword/pattern detection (clinical terms, hollow praise, etc.)
2. Message length (≤ 280 characters)
3. Sentence count (2-4 sentences)
4. Cosine similarity dedup (> 0.85 triggers rejection)
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings


# Forbidden patterns from Section 6.3 of the AIC spec
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    # Clinical language
    (r"\b(therap(y|ist|eutic)|treatment|disorder|symptom\w*|diagnos\w+)\b", "clinical_language"),
    (r"\b(mental\s+health\s+condition|psychiatric|medication|prescri\w+)\b", "clinical_language"),
    (r"\b(psychosis|hallucination\w*|voices?|delusion\w*|schizophreni\w*|bipolar)\b", "clinical_language"),
    (r"\b(illness|disease|patholog\w+|clinical)\b", "clinical_language"),

    # Person/ability attribution
    (r"\byou(?:'re|\s+are)\s+(?:so\s+)?(strong|brave|resilient|amazing|incredible|tough)\b", "person_attribution"),
    (r"\byou\s+can\s+handle\s+anything\b", "person_attribution"),

    # Hollow praise
    (r"\b(great\s+job|awesome|keep\s+it\s+up|you'?re\s+doing\s+(amazing|great|wonderful))\b", "hollow_praise"),
    (r"\b(well\s+done|fantastic|brilliant|superstar|proud\s+of\s+you)\b", "hollow_praise"),

    # Fabricated data
    (r"\b\d+\s*%\b", "fabricated_data"),
    (r"\b(studies?\s+show|research\s+(shows?|suggests?|indicates?)|according\s+to)\b", "fabricated_data"),
    (r"\b(most\s+people|others?\s+like\s+you|many\s+(people|individuals))\b", "fabricated_data"),

    # Minimizing
    (r"\b(it'?s\s+not\s+that\s+bad|just\s+relax|don'?t\s+worry(\s+about\s+it)?)\b", "minimizing"),
    (r"\b(everything\s+will\s+be\s+(fine|okay|ok|alright))\b", "minimizing"),

    # Commands without strategy
    (r"\b(calm\s+down|stop\s+worrying|think\s+positive|cheer\s+up)\b", "command_without_strategy"),
    (r"\b(get\s+over\s+it|snap\s+out\s+of\s+it|pull\s+yourself\s+together)\b", "command_without_strategy"),

    # Time predictions
    (r"\b(this\s+will\s+pass(\s+in)?|you'?ll\s+feel\s+better\s+(soon|tomorrow|in\s+\d+))\b", "time_prediction"),
    (r"\b(it\s+won'?t\s+last|temporary|it'?ll\s+get\s+better)\b", "time_prediction"),
]


class ContentScreener:
    """
    Screens LLM-generated intervention content before delivery.

    All checks must pass for content to be delivered. If content fails,
    the LLM is re-prompted (up to MAX_LLM_RETRIES). If all retries fail,
    a static fallback message is used instead.
    """

    def __init__(
        self,
        max_chars: int = settings.MAX_INTERVENTION_CHARS,
        similarity_threshold: float = settings.COSINE_SIMILARITY_THRESHOLD,
    ):
        self.max_chars = max_chars
        self.similarity_threshold = similarity_threshold

    def screen(self, text: str, recent_texts: list[str] | None = None) -> tuple[bool, list[str]]:
        """
        Screen a message against all content rules.

        Args:
            text: The LLM-generated message to screen.
            recent_texts: Messages delivered to this participant in the past 24 hours.

        Returns:
            Tuple of (passed, list_of_violations).
        """
        violations = []

        # 1. Length check
        if len(text) > self.max_chars:
            violations.append(f"too_long:{len(text)}_chars")

        # 2. Sentence count (split on sentence-ending punctuation)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) < 2:
            violations.append(f"too_few_sentences:{len(sentences)}")
        if len(sentences) > 4:
            violations.append(f"too_many_sentences:{len(sentences)}")

        # 3. Forbidden patterns
        for pattern, category in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"forbidden:{category}")

        # 4. Cosine similarity dedup
        if recent_texts:
            max_sim = self._max_cosine_similarity(text, recent_texts)
            if max_sim > self.similarity_threshold:
                violations.append(f"too_similar:{max_sim:.3f}")

        return (len(violations) == 0, violations)

    def _max_cosine_similarity(self, new_text: str, recent: list[str]) -> float:
        """Compute max cosine similarity between new_text and recent messages."""
        if not recent:
            return 0.0

        corpus = recent + [new_text]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
        return float(sims.max())
