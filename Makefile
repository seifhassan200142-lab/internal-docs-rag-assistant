.PHONY: install run-api run-ui test docker-up docker-down

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ui/streamlit_app.py

test:
	pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down
