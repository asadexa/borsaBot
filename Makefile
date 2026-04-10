.PHONY: install test test-unit test-integration coverage lint infra-up infra-down seed

# ── Install ───────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Tests ─────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short -m "not integration"

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

coverage:
	pytest tests/ -v --cov=borsabot --cov-report=html --cov-report=term-missing
	@echo "Coverage report at htmlcov/index.html"

# ── Lint ──────────────────────────────────────────────────────────
lint:
	ruff check borsabot/ tests/
	mypy borsabot/ --ignore-missing-imports

# ── Infrastructure ────────────────────────────────────────────────
infra-up:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 5
	@docker compose ps

infra-down:
	docker compose down

seed:
	python scripts/seed_db.py

# ── Backtest ──────────────────────────────────────────────────────
backtest:
	python scripts/run_backtest.py --symbol BTCUSDT --days 90
