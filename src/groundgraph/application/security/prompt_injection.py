"""Prompt injection detection for user-provided queries.

Provides a heuristic-based prompt injection detector that flags
suspicious patterns in user queries before they reach the LLM.

Usage:
    from groundgraph.application.security.prompt_injection import (
        PromptInjectionDetector,
        InjectionCheckResult,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

_HIGH_RISK_PATTERNS: list[tuple[str, str]] = [
    ("ignore all previous instructions", "high"),
    ("ignore previous instructions", "high"),
    ("forget all previous rules", "high"),
    ("disregard all prior instructions", "high"),
    ("developer mode", "high"),
    ("dev mode activated", "high"),
    ("enable developer mode", "high"),
    ("[inst]", "medium"),
    ("[/inst]", "medium"),
    ("<svg ", "high"),
    ("<script", "high"),
    ("javascript:", "high"),
    ("${jndi:", "high"),
    ("${env:", "medium"),
    ("{{constructor", "high"),
    ("{{__", "medium"),
    ("<!--", "low"),
    ("--!>", "low"),
]

_MEDIUM_RISK_PATTERNS: list[tuple[str, str]] = [
    ("system:", "medium"),
    ("admin:", "low"),
    ("sudo:", "low"),
    ("<iframe", "medium"),
    ("onerror=", "high"),
    ("onload=", "high"),
]

_ALL_PATTERNS = _HIGH_RISK_PATTERNS + _MEDIUM_RISK_PATTERNS


@dataclass
class InjectionCheckResult:
    """Result of a prompt injection check."""

    is_suspicious: bool
    confidence: float
    matched_rules: list[str]
    risk_level: str

    @property
    def blocked(self) -> bool:
        return self.risk_level == "high"


class PromptInjectionDetector:
    """Heuristic prompt injection detector.

    Checks user queries for common prompt injection patterns:
      - Direct instruction override attempts
      - Role-play / developer mode prompts
      - Special delimiters that may try to escape context
      - Opaque encoded payloads (JNDI, template injection, etc.)

    This is a heuristic fallback. In production, a dedicated LLM-based
    classifier should be used (see ADR-007).
    """

    _patterns: ClassVar[list[tuple[str, str]]] = _ALL_PATTERNS

    def check(self, question: str) -> InjectionCheckResult:
        """Check a user question for prompt injection patterns.

        Returns an InjectionCheckResult with:
          - is_suspicious: True if any pattern matched
          - confidence: 0.0-1.0 based on specificity of match
          - matched_rules: list of matched rule names
          - risk_level: "high", "medium", or "low"
        """
        q_lower = question.lower()
        matched: list[str] = []
        max_risk = "low"

        for pattern, risk in self._patterns:
            if pattern in q_lower:
                matched.append(pattern)
                if risk == "high" and max_risk != "high":
                    max_risk = "high"
                elif risk == "medium" and max_risk == "low":
                    max_risk = "medium"

        is_suspicious = len(matched) > 0
        confidence = min(1.0, len(matched) * 0.3 + 0.5) if is_suspicious else 0.0

        return InjectionCheckResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            matched_rules=matched,
            risk_level=max_risk,
        )


def sanitize_question(question: str) -> str:
    """Sanitize a user question by removing high-risk injection markers.

    This is a best-effort sanitization. High-risk questions should be
    blocked at the `check()` level before reaching this function.
    """
    sanitized = question
    removals = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+=",
        r"${jndi:[^}]*}",
        r"{{constructor[^}]*}}",
    ]
    for pattern in removals:
        sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
    return sanitized
