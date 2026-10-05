# ── Enterprise AI Assistant — Developer Commands ──────────────────────────────
# Run `make help` to list all available targets.

VENV      = venv
PYTHON    = $(VENV)/bin/python
PIP       = $(VENV)/bin/pip
PYTEST    = $(VENV)/bin/pytest
UVICORN   = $(VENV)/bin/uvicorn

.PHONY: help install test test-unit test-integration db-start db-stop dev token clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────

install: ## Create venv and install all Python dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── Tests ─────────────────────────────────────────────────────────────────────

test: ## Run all tests (unit + integration)
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run only unit tests (no I/O, fast)
	$(PYTEST) tests/ -v --tb=short -m unit

test-integration: ## Run only integration tests (mocked LLM/DB)
	$(PYTEST) tests/ -v --tb=short -m integration

# ── Database ──────────────────────────────────────────────────────────────────

db-start: ## Start PostgreSQL + pgvector via Docker Compose
	docker compose up -d db
	@echo "⏳ Waiting for PostgreSQL to be healthy..."
	@until docker compose exec db pg_isready -U postgres -d enterprise_ai; do sleep 1; done
	@echo "✅ Database ready at localhost:5432"

db-stop: ## Stop and remove the database container
	docker compose down

db-shell: ## Open a psql shell inside the running DB container
	docker compose exec db psql -U postgres -d enterprise_ai

db-logs: ## Tail PostgreSQL container logs
	docker compose logs -f db

# ── API Server ────────────────────────────────────────────────────────────────

dev: ## Start FastAPI dev server with hot-reload (requires DB running)
	PYTHONPATH=. $(UVICORN) deployment.main:app \
	  --host 0.0.0.0 --port 8000 --reload --log-level info

# ── Dev Utilities ─────────────────────────────────────────────────────────────

token: ## Generate a dev JWT token (requires server running at :8000)
	@curl -s -X POST http://localhost:8000/api/v1/auth/token \
	  -H "Content-Type: application/json" \
	  -d '{"user_id":"usr_dev_01","tenant_id":"corp_acme_01"}' | python3 -m json.tool

health: ## Check API health endpoints
	@echo "── Liveness ──"
	@curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
	@echo "── Readiness (DB) ──"
	@curl -s http://localhost:8000/api/v1/health/db | python3 -m json.tool

clean: ## Remove venv, caches, bytecode
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
