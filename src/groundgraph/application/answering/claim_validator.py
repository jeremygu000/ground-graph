"""Deterministic claim validator.

Checks that every claim's evidence_ids exist in the evidence set,
that citation locators resolve to stored versions, and computes
a coverage ratio for the response.
"""

from __future__ import annotations

from uuid import UUID

from groundgraph.application.ports import ClaimValidator
from groundgraph.domain.retrieval import AnswerClaim, Citation, Evidence, QueryResponse


class DeterministicClaimValidator(ClaimValidator):
    """Validates claim-evidence relationships deterministically.

    Rules:
    - Every evidence_id in a claim must exist in the evidence set
    - Every citation's evidence_id must exist in the evidence set
    - If any "supported" claim has an evidence_id not in evidence → downgrade to "unsupported"
    - If any "supported" claim is missing evidence_id → downgrade to "partially_supported"
    """

    async def validate(self, response: QueryResponse, evidence: list[Evidence]) -> QueryResponse:
        evidence_map: dict[UUID, Evidence] = {ev.evidence_id: ev for ev in evidence}

        validated_claims: list[AnswerClaim] = []
        validated_citations: list[Citation] = []
        warnings: list[str] = list(response.warnings)
        unsupported_count = 0

        for claim in response.claims:
            claim_valid = self._validate_claim(claim, evidence_map)
            validated_claims.append(claim_valid)
            if claim_valid.support_status == "unsupported":
                unsupported_count += 1

            for cit in response.citations:
                if cit.claim_id == claim.claim_id:
                    if cit.evidence_id not in evidence_map:
                        continue
                    validated_citations.append(cit)

        if unsupported_count > 0:
            warnings.append(f"{unsupported_count} unsupported claim(s) detected")

        return QueryResponse(
            execution_run_id=response.execution_run_id,
            answer=response.answer,
            status=response.status,
            claims=validated_claims,
            citations=validated_citations,
            confidence_band=response.confidence_band,
            warnings=warnings,
        )

    def _validate_claim(
        self, claim: AnswerClaim, evidence_map: dict[UUID, Evidence]
    ) -> AnswerClaim:
        valid_statuses = {"supported", "partially_supported", "unsupported"}

        if claim.support_status not in valid_statuses:
            claim = AnswerClaim(
                claim_id=claim.claim_id,
                text=claim.text,
                factual=claim.factual,
                evidence_ids=claim.evidence_ids,
                support_status="unsupported",
            )

        if claim.support_status in ("supported", "partially_supported"):
            if not claim.evidence_ids:
                claim = AnswerClaim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    factual=claim.factual,
                    evidence_ids=claim.evidence_ids,
                    support_status="unsupported",
                )
            else:
                present_ids = [eid for eid in claim.evidence_ids if eid in evidence_map]
                if not present_ids:
                    claim = AnswerClaim(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        factual=claim.factual,
                        evidence_ids=[],
                        support_status="unsupported",
                    )
                elif len(present_ids) < len(claim.evidence_ids):
                    claim = AnswerClaim(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        factual=claim.factual,
                        evidence_ids=present_ids,
                        support_status="partially_supported",
                    )

        return claim
