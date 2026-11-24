.PHONY: up down build logs status test clean

up:
	docker compose up --build -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

status:
	docker compose ps

test:
	python -m pytest -v backend/tests

clean:
	docker compose down -v
