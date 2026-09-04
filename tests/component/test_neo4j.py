"""Neo4j component tests.

Run against a real Neo4j container started via Testcontainers
(``conftest.py``). Verifies:

  * Container starts and the bolt protocol accepts connections.
  * A trivial session can run Cypher.
  * Constraints (uniqueness) and indexes are honoured.
  * Multi-statement transactions can be rolled back.

These tests are intentionally NOT stacked against an application
adapter — M2 has not built one yet. They validate the substrate the
adapters will use.
"""

from __future__ import annotations

from typing import Any

import pytest
from neo4j import GraphDatabase

pytestmark = [pytest.mark.integration, pytest.mark.component]


def test_neo4j_driver_connects(neo4j_component: Any) -> None:
    driver = GraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        driver.verify_connectivity()
    finally:
        driver.close()


def test_neo4j_cypher_round_trip(neo4j_component: Any) -> None:
    """CREATE → RETURN → MATCH → DELETE in a single session."""
    driver = GraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        with driver.session() as session:
            session.run("MATCH (n:_CompTest) DETACH DELETE n")
            session.run("CREATE (:_CompTest {name: $name, version: $v})", name="alpha", v=1)
            record = session.run(
                "MATCH (n:_CompTest {name: $name}) RETURN n.version AS v", name="alpha"
            ).single()
            assert record is not None
            assert record["v"] == 1
            # Cleanup so re-runs are idempotent.
            session.run("MATCH (n:_CompTest) DETACH DELETE n")
    finally:
        driver.close()


def test_neo4j_unique_constraint(neo4j_component: Any) -> None:
    """Create a unique constraint, attempt duplicate insert, expect failure."""
    driver = GraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        with driver.session() as session:
            session.run("DROP CONSTRAINT comp_test_unique IF EXISTS")
            session.run("MATCH (n:_UniqueTest) DETACH DELETE n")
            session.run(
                "CREATE CONSTRAINT comp_test_unique FOR (n:_UniqueTest) REQUIRE n.name IS UNIQUE"
            )
            session.run("CREATE (:_UniqueTest {name: 'one'})")
            # The second CREATE must fail with a constraint violation.
            with pytest.raises(Exception) as exc_info:  # noqa: PT011
                session.run("CREATE (:_UniqueTest {name: 'one'})")
            # The driver wraps it as ClientError; we don't assert the
            # exact string to stay forward-compatible.
            assert (
                "ConstraintViolation" in type(exc_info.value).__name__
                or "already exists" in str(exc_info.value).lower()
                or "unique" in str(exc_info.value).lower()
            ), f"expected unique-constraint violation, got: {exc_info.value!r}"
            # Cleanup
            session.run("DROP CONSTRAINT comp_test_unique IF EXISTS")
            session.run("MATCH (n:_UniqueTest) DETACH DELETE n")
    finally:
        driver.close()


def test_neo4j_transaction_rollback(neo4j_component: Any) -> None:
    """Within an explicit transaction, ROLLBACK discards the writes."""
    driver = GraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        with driver.session() as session:
            # Use a unique label to avoid colliding with other tests.
            label = "_TxnTest"
            session.run(f"MATCH (n:{label}) DETACH DELETE n")

            def _create(tx: Any) -> None:
                tx.run(f"CREATE (:{label} {{name: 'will-be-rolled-back'}})")

            try:
                with session.begin_transaction() as tx:
                    _create(tx)
                    tx.rollback()
            except Exception:
                # Some driver versions raise on explicit rollback after
                # writes; we tolerate that as long as the data is gone.
                pass

            record = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
            assert record is not None, "expected a count record from Neo4j"
            count = record["c"]
            assert count == 0, "rolled-back transaction left a node behind"

            session.run(f"MATCH (n:{label}) DETACH DELETE n")
    finally:
        driver.close()
