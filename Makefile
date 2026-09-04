# Makefile for Agentic GraphRAG

# Always run from project root
ROOT := $(shell pwd)
PY  := uv run python
PIP := uv pip

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install dependencies and pre-commit
	uv sync --all-extras
	uv run pre-commit install || true

.PHONY: install
install: ## Install runtime dependencies only
	uv sync

.PHONY: format
format: ## Run ruff formatter
	uv run ruff format .

.PHONY: lint
lint: ## Run ruff linter
	uv run ruff check . --fix

.PHONY: lint-check
lint-check: ## Run ruff linter (no fix)
	uv run ruff check .

.PHONY: typecheck
typecheck: ## Run pyright + mypy on core layers
	uv run pyright
	uv run mypy

.PHONY: pyright
pyright: ## Run pyright only
	uv run pyright

.PHONY: mypy
mypy: ## Run mypy on domain + application (strict)
	uv run mypy

.PHONY: test
test: ## Run unit tests
	uv run pytest -q -m "not integration and not e2e"

.PHONY: test-integration
test-integration: ## Run integration tests (requires Docker)
	uv run pytest -q -m "integration"

.PHONY: test-all
test-all: ## Run all tests including integration
	uv run pytest -q

.PHONY: test-cov
test-cov: ## Run unit tests with coverage
	uv run pytest -q -m "not integration and not e2e" --cov=src/graphrag --cov-report=term-missing

.PHONY: eval-smoke
eval-smoke: ## Run evaluation smoke tests
	uv run python -m graphrag.application.evaluation.smoke

.PHONY: check
check: format lint-check typecheck test ## Run all quality gates

.PHONY: fix
fix: ## Auto-fix what we can (format + lint --fix), then re-check
	uv run ruff format .
	uv run ruff check . --fix
	$(MAKE) check

.PHONY: ci
ci: lint-check typecheck test ## CI pipeline (no format)

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .pyright .mypy_cache
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: docs-build
docs-build: ## Build docs (placeholder)
	@echo "No docs build configured yet"

.PHONY: adr-new
adr-new: ## Create new ADR (usage: make adr-new NAME=002-storage)
	@if [ -z "$(NAME)" ]; then echo "Usage: make adr-new NAME=002-short-title"; exit 1; fi
	@if [ -z "$(TITLE)" ]; then echo "Usage: make adr-new TITLE='Title'"; exit 1; fi
	@N=$$(echo $(NAME) | cut -d- -f1); \
	 FILE="docs/adr/ADR-$$N-$(NAME).md"; \
	 cp docs/adr/ADR-000-template.md $$FILE; \
	 sed -i '' "s/ADR-000/ADR-$$N/" $$FILE; \
	 sed -i '' "s/\[Short Title\]/$(TITLE)/" $$FILE; \
	 echo "Created $$FILE"

# -----------------------------------------------------------------------------
# Local infrastructure (Docker Compose stack)
# -----------------------------------------------------------------------------
COMPOSE := docker compose
COMPOSE_FILE := docker-compose.yml

.PHONY: infra-up
infra-up: ## Start all local infrastructure services (detached)
	$(COMPOSE) -f $(COMPOSE_FILE) --env-file .env up -d
	@$(COMPOSE) -f $(COMPOSE_FILE) ps

.PHONY: infra-down
infra-down: ## Stop all local infrastructure services
	$(COMPOSE) -f $(COMPOSE_FILE) down

.PHONY: infra-down-volumes
infra-down-volumes: ## Stop services and remove all volumes (destructive)
	$(COMPOSE) -f $(COMPOSE_FILE) down -v

.PHONY: infra-logs
infra-logs: ## Tail logs from all services
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f --tail=100

.PHONY: infra-ps
infra-ps: ## Show running services and health
	$(COMPOSE) -f $(COMPOSE_FILE) ps

.PHONY: infra-restart
infra-restart: ## Restart all services
	$(COMPOSE) -f $(COMPOSE_FILE) restart

.PHONY: infra-validate
infra-validate: ## Validate the docker-compose file
	$(COMPOSE) -f $(COMPOSE_FILE) config -q
