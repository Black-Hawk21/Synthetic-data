.PHONY: install install-frontend backend frontend generate-data train evaluate red-team closed-loop test docker-up docker-down

install:
	pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

backend:
	uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

generate-data:
	python scripts/generate_data.py --n 100000

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

red-team:
	python scripts/run_red_team.py --list

closed-loop:
	python scripts/run_closed_loop.py --n 20000 --iterations 3

test:
	pytest tests/ -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down
