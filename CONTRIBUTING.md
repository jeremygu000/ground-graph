# Contributing to Agentic GraphRAG

Thank you for your interest in contributing. This project follows a strict
milestone-driven plan documented in [`docs/plan.md`](docs/plan.md) and the
operational rules in [`AGENTS.md`](AGENTS.md). All contributions must
respect those documents.

By participating you agree to the Apache License 2.0 (see [`LICENSE`](LICENSE))
for any contribution you submit.

## Ground rules

1. **One milestone at a time.** Do not jump ahead or smuggle unrelated
   work into a milestone. See `docs/plan.md` §0.3 for the current ledger.
2. **Respect the dependency direction.**

   ```
   domain  <-  application  <-  workflows / API  <-  infrastructure composition
   ```

   Domain and application layers **must not** import LangGraph, FastAPI,
   OpenAI, Neo4j, SQLAlchemy, Phoenix, or any framework. A test in
   `tests/unit/test_domain_isolation.py` enforces this in CI.
3. **No shortcuts on quality gates.** A milestone is not complete until
   `make check` (format, lint, typecheck, unit tests) passes locally and
   CI is green.
4. **No mock-and-pray.** Tests must not be skipped, mocked out, or left
   failing while claiming success. See `docs/plan.md` §0.1 rule 4.
5. **No hidden chain-of-thought, no raw secrets, no PII in telemetry.**
   See `docs/plan.md` §0.1 rules 6–8 and §10.3.
6. **Authorization is pre-retrieval.** Never let unfiltered results
   reach a model context. See `docs/plan.md` §0.1 rule 9 and §12.1.

## Development environment

Requirements: Python 3.12+, [`uv`](https://github.com/astral-sh/uv),
Docker, Docker Compose.

```bash
git clone <your-fork>
cd ground-graph
make setup         # uv sync --all-extras
make infra-up      # start Postgres/pgvector, Neo4j, MinIO, Phoenix, OTel, Prometheus, Grafana
make check         # format + lint + typecheck + unit tests
make test-integration   # runs against the docker stack
```

## Code style

- Formatter and linter: [Ruff](https://docs.astral.sh/ruff/).
- Type checker: [Pyright](https://microsoft.github.io/pyright/) in
  `standard` mode globally, `strict` on `src/graphrag/domain` and
  `src/graphrag/application`.
- Tests: `pytest` with `pytest-asyncio` in `auto` mode.
- Public APIs and Pydantic contracts must have docstrings.
- No comments unless they explain non-obvious "why" (system instruction).

## Pull request workflow

1. Open an issue describing the problem before opening a PR for
   non-trivial work. Reference the milestone and acceptance criteria
   from `docs/plan.md` that the PR satisfies.
2. Create a topic branch: `git checkout -b m<N>-<short-slug>`.
3. Implement, test, and run `make check`.
4. If your change introduces a new architectural decision, add an ADR:
   `cp docs/adr/ADR-000-template.md docs/adr/ADR-<NNN>-<slug>.md` and
   fill it in. See existing ADRs in `docs/adr/` for tone.
5. Update the progress ledger in `docs/plan.md` if you complete a
   milestone (the rule is one `[ ]` → `[x]` per acceptance pass).
6. Write a PR description that reports, per `AGENTS.md` §13:
   - milestone and tasks completed;
   - files and migrations changed;
   - architecture or ADR decisions;
   - commands executed and their results;
   - evaluation impact (or "not applicable" with reason);
   - known limitations or failing cases;
   - the next smallest executable task.
7. All CI checks must pass. Reviewers will not merge a red PR.

## Commit messages

- Imperative mood, present tense: "Add", not "Added".
- First line ≤ 72 characters, no trailing period.
- Reference the milestone: `M0:`, `M2:`, `M7:`.
- For ADRs use `ADR-NNN: <title>`.

Examples:

```
M0: initialize uv project with strict quality gates
M2: add reified-fact domain contracts
ADR-002: hybrid Postgres + Neo4j storage
```

## Tests

- Unit tests live in `tests/unit/` and run on every PR.
- Integration tests in `tests/integration/` require the Docker stack
  to be up (`make infra-up`) and are tagged `@pytest.mark.integration`.
- Adversarial cases (prompt injection, authorization) live in
  `tests/adversarial/`. Add new ones as you discover attack vectors.
- Coverage target: 85% line coverage on `src/graphrag`, but never
  game the number — direct tests for security, authorization, and
  provenance take priority.

## Adding dependencies

1. Add the package to the correct group in `pyproject.toml`
   (`dependencies`, `dev`, `langgraph`, or `ingestion`).
2. Run `uv sync` and commit the updated `uv.lock`.
3. Never use floating version constraints in CI. Pin minor versions
   or use `>=X.Y,<X+1`.
4. Justify the dependency in the PR description: what layer uses it,
   what port it implements, and which framework guard (if any) it
   affects.

## Security

- Never commit secrets, API keys, tokens, or production credentials.
  `.env` is git-ignored. The CI secret-scan job will fail a PR that
  looks like it contains one.
- If you accidentally commit a secret, rotate it immediately —
  deleting the line is not enough.
- Report security issues privately to the maintainers listed in
  `CODEOWNERS` (when present) before opening a public issue.

## Code of conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
Be respectful, assume good faith, and focus on the work. Maintainers may
remove comments or PRs that violate this standard.

## Questions?

Open a GitHub issue with the `question` label, or check `docs/plan.md`
and `AGENTS.md` first — most "how do I…" questions are answered there.
