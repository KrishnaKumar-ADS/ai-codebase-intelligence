.PHONY: dev test lint migrate clean

dev:
	docker compose -f docker/docker-compose.yml up --build

test:
	cd backend && pytest tests/ -v

lint:
	cd backend && ruff check . && mypy .

migrate:
	cd backend && alembic upgrade head

clean:
	docker compose -f docker/docker-compose.yml down -v
