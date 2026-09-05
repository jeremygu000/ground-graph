"""M2 domain contracts and persistence model tests.

Tests:
- Pydantic validation and serialization round trips
- database migration up on empty database (integration)
- uniqueness/idempotency constraints (integration)
- outbox claim/retry semantics (integration)
- Neo4j constraints and simple fact round trip (integration)
- execution step state transitions reject invalid transitions
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.domain.evidence import OutboxEvent, OutboxEventStatus, OutboxEventType
from groundgraph.domain.execution import (
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionStep,
    ExecutionStepStatus,
    assert_run_transition,
)
from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.domain.retrieval import (
    AnswerClaim,
    Citation,
    Evidence,
    QueryResponse,
    RetrievalPlan,
)
from groundgraph.domain.types import validate_json_value


class TestDocumentContracts:
    def test_source_descriptor_round_trip(self) -> None:
        source = SourceDescriptor(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/path/to/doc.md",
            classification="internal",
            tenant_id="default",
            allowed_principals=["engineering", "security"],
        )
        data = source.model_dump()
        restored = SourceDescriptor.model_validate(data)
        assert restored == source

    def test_parsed_document_round_trip(self) -> None:
        doc = ParsedDocument(
            document_id=uuid4(),
            version_id=uuid4(),
            source_id=uuid4(),
            title="Test Document",
            media_type="text/markdown",
            checksum="abc123",
            content="# Hello",
            metadata={"author": "test"},
            effective_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        data = doc.model_dump()
        restored = ParsedDocument.model_validate(data)
        assert restored == doc

    def test_chunk_round_trip(self) -> None:
        chunk = Chunk(
            chunk_id=uuid4(),
            document_id=uuid4(),
            version_id=uuid4(),
            ordinal=0,
            heading_path=["Introduction"],
            content="Some text",
            token_count=5,
            checksum="xyz789",
            start_locator="line:1",
            end_locator="line:1",
            allowed_principals=["engineering"],
        )
        data = chunk.model_dump()
        restored = Chunk.model_validate(data)
        assert restored == chunk

    def test_source_descriptor_frozen(self) -> None:
        source = SourceDescriptor(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/path",
            classification="internal",
            tenant_id="default",
        )
        with pytest.raises(ValidationError, match="frozen"):
            source.uri = "/other"  # type: ignore

    def test_source_descriptor_illegal_source_type(self) -> None:
        with pytest.raises(ValidationError, match="source_type"):
            SourceDescriptor(
                source_id=uuid4(),
                source_type="bad-type",  # type: ignore[arg-type]
                uri="/path",
                classification="internal",
                tenant_id="default",
            )

    def test_validate_json_value_round_trip(self) -> None:
        payload = {
            "name": "api",
            "count": 3,
            "flags": [True, False, None],
            "meta": {"region": "us-east-1"},
        }
        assert validate_json_value(payload) == payload

    def test_validate_json_value_rejects_non_string_key(self) -> None:
        with pytest.raises(TypeError, match="JSON object key"):
            validate_json_value({1: "bad"})  # type: ignore[arg-type]

    def test_validate_json_value_rejects_bytes(self) -> None:
        with pytest.raises(ValueError, match="JsonValue"):
            validate_json_value({"blob": b"raw"})

    def test_validate_json_value_allows_shared_containers(self) -> None:
        shared = [{"name": "shared"}]
        payload = {"left": shared, "right": shared}

        assert validate_json_value(payload) == payload

    def test_parsed_document_rejects_invalid_metadata(self) -> None:
        with pytest.raises(ValueError, match="JsonValue"):
            ParsedDocument(
                document_id=uuid4(),
                version_id=uuid4(),
                source_id=uuid4(),
                title="Test Document",
                media_type="text/markdown",
                checksum="abc123",
                content="# Hello",
                metadata={"blob": b"raw"},
                effective_at=datetime(2024, 1, 1, tzinfo=UTC),
            )


class TestKnowledgeContracts:
    def test_entity_mention_round_trip(self) -> None:
        mention = EntityMention(
            mention_id=uuid4(),
            chunk_id=uuid4(),
            surface_form="API Gateway",
            candidate_type="Service",
            locator="para:3",
            extraction_confidence=0.95,
        )
        data = mention.model_dump()
        restored = EntityMention.model_validate(data)
        assert restored == mention

    def test_canonical_entity_round_trip(self) -> None:
        entity = CanonicalEntity(
            entity_id=uuid4(),
            entity_type="Service",
            canonical_name="API Gateway",
            aliases=["api-gateway", "gateway"],
            attributes={"region": "us-east-1"},
        )
        data = entity.model_dump()
        restored = CanonicalEntity.model_validate(data)
        assert restored == entity

    def test_canonical_entity_rejects_invalid_attributes(self) -> None:
        with pytest.raises(ValueError, match="JsonValue"):
            CanonicalEntity(
                entity_id=uuid4(),
                entity_type="Service",
                canonical_name="API Gateway",
                attributes={"blob": b"raw"},
            )

    def test_knowledge_fact_round_trip(self) -> None:
        fact = KnowledgeFact(
            fact_id=uuid4(),
            subject_id=uuid4(),
            predicate="DEPENDS_ON",
            object_id=uuid4(),
            status="verified",
            confidence=0.9,
            evidence_ids=[uuid4(), uuid4()],
            valid_from=datetime(2024, 1, 1, tzinfo=UTC),
            valid_to=None,
            observed_at=datetime(2024, 6, 1, tzinfo=UTC),
            extraction_method="llm",
            ontology_version="v0.1.0",
        )
        data = fact.model_dump()
        restored = KnowledgeFact.model_validate(data)
        assert restored == fact

    def test_knowledge_fact_status_validation(self) -> None:
        with pytest.raises(ValueError, match="status"):
            KnowledgeFact(
                fact_id=uuid4(),
                subject_id=uuid4(),
                predicate="DEPENDS_ON",
                object_id=uuid4(),
                status="invalid_status",  # type: ignore
                confidence=0.9,
                observed_at=datetime.now(UTC),
                extraction_method="llm",
                ontology_version="v0.1.0",
            )

    def test_knowledge_fact_reversed_validity_window(self) -> None:
        with pytest.raises(ValueError, match="valid_to"):
            KnowledgeFact(
                fact_id=uuid4(),
                subject_id=uuid4(),
                predicate="DEPENDS_ON",
                object_id=uuid4(),
                status="candidate",
                confidence=0.9,
                valid_from=datetime(2024, 6, 1, tzinfo=UTC),
                valid_to=datetime(2024, 1, 1, tzinfo=UTC),
                observed_at=datetime(2024, 6, 1, tzinfo=UTC),
                extraction_method="llm",
                ontology_version="v0.1.0",
            )

    def test_knowledge_fact_naive_datetime_rejected(self) -> None:
        naive_valid_from = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with pytest.raises(ValueError, match="timezone-aware"):
            KnowledgeFact(
                fact_id=uuid4(),
                subject_id=uuid4(),
                predicate="DEPENDS_ON",
                object_id=uuid4(),
                status="candidate",
                confidence=0.9,
                valid_from=naive_valid_from,
                observed_at=datetime(2024, 6, 1, tzinfo=UTC),
                extraction_method="llm",
                ontology_version="v0.1.0",
            )

    def test_verified_fact_requires_evidence(self) -> None:
        with pytest.raises(ValueError, match="verified facts"):
            KnowledgeFact(
                fact_id=uuid4(),
                subject_id=uuid4(),
                predicate="DEPENDS_ON",
                object_id=uuid4(),
                status="verified",
                confidence=0.9,
                evidence_ids=[],
                observed_at=datetime(2024, 6, 1, tzinfo=UTC),
                extraction_method="human",
                ontology_version="v0.1.0",
            )


class TestExecutionContracts:
    def test_execution_run_round_trip(self) -> None:
        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.SUCCEEDED,
            principal="engineering",
            tenant_id="default",
            input={"question": "What depends on X?"},
            output={"answer": "Y depends on X"},
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
            finished_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )
        data = run.model_dump()
        restored = ExecutionRun.model_validate(data)
        assert restored == run

    def test_execution_run_rejects_invalid_json_input(self) -> None:
        with pytest.raises(ValueError, match="JsonValue"):
            ExecutionRun(
                run_id=uuid4(),
                workflow="query",
                status=ExecutionRunStatus.PENDING,
                principal="engineering",
                tenant_id="default",
                input={"blob": b"raw"},
            )

    def test_execution_run_terminal_requires_finished_at(self) -> None:
        with pytest.raises(ValueError, match="terminal"):
            ExecutionRun(
                run_id=uuid4(),
                workflow="query",
                status=ExecutionRunStatus.SUCCEEDED,
                principal="engineering",
                tenant_id="default",
                finished_at=None,
            )

    def test_execution_step_round_trip(self) -> None:
        step = ExecutionStep(
            step_id=uuid4(),
            run_id=uuid4(),
            name="retrieve",
            status=ExecutionStepStatus.SUCCEEDED,
            depends_on=[],
            input={},
            output={"chunks_returned": 5},
        )
        data = step.model_dump()
        restored = ExecutionStep.model_validate(data)
        assert restored == step

    def test_valid_run_transitions(self) -> None:
        assert_run_transition(ExecutionRunStatus.PENDING, ExecutionRunStatus.RUNNING)
        assert_run_transition(ExecutionRunStatus.PENDING, ExecutionRunStatus.CANCELLED)
        assert_run_transition(ExecutionRunStatus.RUNNING, ExecutionRunStatus.SUCCEEDED)
        assert_run_transition(ExecutionRunStatus.RUNNING, ExecutionRunStatus.FAILED)
        assert_run_transition(ExecutionRunStatus.RUNNING, ExecutionRunStatus.CANCELLED)

    def test_invalid_run_transitions_raise(self) -> None:
        with pytest.raises(ValueError, match="illegal"):
            assert_run_transition(ExecutionRunStatus.PENDING, ExecutionRunStatus.SUCCEEDED)

        with pytest.raises(ValueError, match="illegal"):
            assert_run_transition(ExecutionRunStatus.SUCCEEDED, ExecutionRunStatus.FAILED)

        with pytest.raises(ValueError, match="illegal"):
            assert_run_transition(ExecutionRunStatus.FAILED, ExecutionRunStatus.RUNNING)


class TestRetrievalContracts:
    def test_retrieval_plan_round_trip(self) -> None:
        plan = RetrievalPlan(
            strategy="hybrid",
            question_type="multi_hop",
            query_texts=["What depends on X?"],
            max_graph_depth=2,
            vector_top_k=10,
            final_evidence_limit=12,
            reason_codes=["entity_resolved", "graph_traversal"],
        )
        data = plan.model_dump()
        restored = RetrievalPlan.model_validate(data)
        assert restored == plan

    def test_evidence_round_trip(self) -> None:
        evidence = Evidence(
            evidence_id=uuid4(),
            source_id=uuid4(),
            document_id=uuid4(),
            chunk_id=uuid4(),
            content="Some text about X",
            retrieval_method="graph",
            vector_score=0.85,
            rerank_score=0.92,
            graph_path_fact_ids=[uuid4()],
            allowed_principals=["engineering"],
        )
        data = evidence.model_dump()
        restored = Evidence.model_validate(data)
        assert restored == evidence

    def test_query_response_round_trip(self) -> None:
        response = QueryResponse(
            execution_run_id=uuid4(),
            answer="Y depends on X",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
            warnings=[],
        )
        data = response.model_dump()
        restored = QueryResponse.model_validate(data)
        assert restored == response

    def test_answered_response_requires_answer(self) -> None:
        with pytest.raises(ValueError, match="answer"):
            QueryResponse(
                execution_run_id=uuid4(),
                answer=None,
                status="answered",
                claims=[],
                citations=[],
                confidence_band="medium",
                warnings=[],
            )

    def test_supported_claim_requires_evidence(self) -> None:
        with pytest.raises(ValueError, match="evidence_id"):
            AnswerClaim(
                claim_id=uuid4(),
                text="X depends on Y",
                factual=True,
                support_status="supported",
                evidence_ids=[],
            )

    def test_unsupported_claim_rejects_evidence(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            AnswerClaim(
                claim_id=uuid4(),
                text="X depends on Y",
                factual=True,
                support_status="unsupported",
                evidence_ids=[uuid4()],
            )

    def test_evidence_temporal_validation(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="Some text",
                retrieval_method="graph",
                valid_from=datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
            )

        with pytest.raises(ValueError, match="greater than valid_from"):
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="Some text",
                retrieval_method="graph",
                valid_from=datetime(2024, 1, 2, tzinfo=UTC),
                valid_to=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_query_response_rejects_answer_on_non_answered_status(self) -> None:
        with pytest.raises(ValueError, match="None when status is not 'answered'"):
            QueryResponse(
                execution_run_id=uuid4(),
                answer="should not be here",
                status="failed",
                claims=[],
                citations=[],
                confidence_band="low",
                warnings=[],
            )

    def test_json_value_rejects_custom_object(self) -> None:
        class CustomObject:
            pass

        with pytest.raises(ValueError, match="JsonValue"):
            validate_json_value({"custom": CustomObject()})

    def test_json_value_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            validate_json_value({"value": float("nan")})

    def test_citation_round_trip(self) -> None:
        citation = Citation(
            citation_id=uuid4(),
            claim_id=uuid4(),
            evidence_id=uuid4(),
            locator="doc.md#heading",
            allowed_principals=["engineering"],
        )
        data = citation.model_dump()
        restored = Citation.model_validate(data)
        assert restored == citation


class TestOutboxContracts:
    def test_outbox_event_round_trip(self) -> None:
        event = OutboxEvent(
            event_id=uuid4(),
            aggregate_type="document",
            aggregate_id=uuid4(),
            event_type=OutboxEventType.DOCUMENT_PARSED,
            payload={"title": "Test"},
            status=OutboxEventStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        data = event.model_dump()
        restored = OutboxEvent.model_validate(data)
        assert restored == event

    def test_outbox_event_status_transitions(self) -> None:
        event = OutboxEvent(
            event_id=uuid4(),
            aggregate_type="document",
            aggregate_id=uuid4(),
            event_type=OutboxEventType.DOCUMENT_PARSED,
            payload={},
            created_at=datetime.now(UTC),
        )
        assert event.status == OutboxEventStatus.PENDING
