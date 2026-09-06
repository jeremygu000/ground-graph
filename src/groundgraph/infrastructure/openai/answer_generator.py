"""Evidence-only answer generator using structured output."""

from __future__ import annotations

from uuid import uuid4

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from groundgraph.application.ports import AnswerGenerator
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.retrieval import AnswerClaim, Citation, QueryResponse, RetrievalPlan


class _ClaimOutput(BaseModel):
    text: str
    factual: bool
    support_status: str
    evidence_ids: list[str]


class _ResponseOutput(BaseModel):
    answer: str
    claims: list[_ClaimOutput]


SYSTEM_PROMPT = (
    "You are a factual question-answering assistant. Your role is to answer questions "
    "based ONLY on the provided evidence. You must cite every factual claim.\n\n"
    "Rules:\n"
    "1. Answer using ONLY the evidence provided.\n"
    "2. If the evidence does not support an answer, say 'I don't know'.\n"
    "3. Every factual claim must be cited using [citation_id] markers.\n"
    "4. If you cannot determine the answer, respond with status 'insufficient_evidence'.\n"
    "5. Be concise and precise."
)

CLAIM_EVIDENCE_PROMPT = (
    "For each claim in the answer, list the evidence_ids that support it. "
    "Evidence IDs: {evidence_ids}"
)


class EvidenceOnlyAnswerGenerator(AnswerGenerator):
    """Answer generator that only uses retrieved evidence, with structured output."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._client = client or AsyncOpenAI(api_key=cfg.openai_api_key_value)
        self._model = model or cfg.generation_model

    async def generate(
        self,
        question: str,
        evidence: list,
        retrieval_plan: RetrievalPlan,
    ) -> QueryResponse:
        if not evidence:
            return QueryResponse(
                execution_run_id=uuid4(),
                answer=None,
                status="insufficient_evidence",
                claims=[],
                citations=[],
                confidence_band="low",
                warnings=["No evidence retrieved"],
            )

        evidence_context = self._build_evidence_context(evidence)

        try:
            response = await self._call_llm(question, evidence_context, evidence)
        except Exception:
            return QueryResponse(
                execution_run_id=uuid4(),
                answer=None,
                status="failed",
                claims=[],
                citations=[],
                confidence_band="low",
                warnings=["Answer generation failed"],
            )

        return self._build_response(response, evidence)

    def _build_evidence_context(self, evidence: list) -> str:
        parts = []
        for ev in evidence:
            parts.append(f"[{ev.evidence_id}] {ev.content}")
        return "\n\n".join(parts)

    async def _call_llm(
        self,
        question: str,
        evidence_context: str,
        evidence: list,
    ) -> _ResponseOutput:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Evidence:\n{evidence_context}\n\n"
                    "Provide your answer in JSON format with 'answer', and 'claims' array."
                ),
            },
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw = response.choices[0].message.content or "{}"
        return _ResponseOutput.model_validate_json(raw)

    def _build_response(
        self,
        output: _ResponseOutput,
        evidence: list,
    ) -> QueryResponse:
        evidence_map = {str(ev.evidence_id): ev for ev in evidence}

        claims: list[AnswerClaim] = []
        citations: list[Citation] = []
        claim_id = uuid4()

        valid_support_status = {"supported", "partially_supported", "unsupported"}

        for claim_output in output.claims:
            evidence_ids: list[str] = claim_output.evidence_ids or []
            mapped_ids = [
                evidence_map[cid].evidence_id for cid in evidence_ids if cid in evidence_map
            ]

            raw_status = claim_output.support_status or "unsupported"
            if raw_status not in valid_support_status:
                raw_status = "unsupported"

            # Fail-closed: LLM may fabricate evidence IDs that don't exist in our
            # evidence set.  If a claim is "supported" but maps to no real evidence,
            # we must downgrade it to "unsupported" — otherwise the domain
            # AnswerClaim validator raises.  We treat this as a grounding failure
            # of the whole response.
            if raw_status in ("supported", "partially_supported") and not mapped_ids:
                raw_status = "unsupported"

            try:
                claim = AnswerClaim(
                    claim_id=claim_id,
                    text=claim_output.text,
                    factual=claim_output.factual,
                    evidence_ids=mapped_ids,
                    support_status=raw_status,  # type: ignore[arg-type]
                )
            except Exception:
                # Validation failure (e.g. status/evidence mismatch after
                # downgrading).  Skip this claim and continue.
                claim_id = uuid4()
                continue

            claims.append(claim)

            for evidence_id in mapped_ids:
                ev = evidence_map[str(evidence_id)]
                chunk_locator = (
                    f"{ev.chunk_id}@{ev.version_id}"
                    if ev.chunk_id is not None and ev.version_id is not None
                    else str(ev.structured_record_id or ev.evidence_id)
                )
                citation = Citation(
                    citation_id=uuid4(),
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    locator=chunk_locator,
                    allowed_principals=ev.allowed_principals,
                )
                citations.append(citation)

            claim_id = uuid4()

        supported_claim_count = sum(
            1 for c in claims if c.support_status in ("supported", "partially_supported")
        )
        high_confidence_threshold = 3

        # Fail closed: if the model claims an answer but produced no supported
        # claims, the answer has no grounding — refuse rather than leak an
        # unsupported assertion.
        if output.answer and output.answer.strip():
            if supported_claim_count == 0:
                return QueryResponse(
                    execution_run_id=uuid4(),
                    answer=None,
                    status="insufficient_evidence",
                    claims=[],
                    citations=[],
                    confidence_band="low",
                    warnings=["LLM answer had no supported claims"],
                )
            status = "answered"
            confidence = "high" if supported_claim_count >= high_confidence_threshold else "medium"
        else:
            status = "insufficient_evidence"
            confidence = "low"

        return QueryResponse(
            execution_run_id=uuid4(),
            answer=output.answer if output.answer else None,
            status=status,
            claims=claims,
            citations=citations,
            confidence_band=confidence,
        )

    async def close(self) -> None:
        await self._client.close()
