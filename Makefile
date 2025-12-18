```makefile
.PHONY: venv install data features train eval tune explain api docker-build docker-run docker-run-thresh clean

PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip

venv:
	$(PY) -m venv $(VENV)

install:
	$(PIP) install -r requirements.txt

data:
	$(PY) -m src.ingestion.make_dataset

features:
	$(PY) -m src.features.build_features

train:
	$(PY) -m src.models.train

eval:
	$(PY) -m src.eval.evaluate

tune:
	$(PY) -m src.eval.threshold_tuning

explain:
	$(PY) -m src.eval.explain

api:
	$(VENV)/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t incident-ml-classifier:latest .

docker-run:
	docker run -p 8001:8000 incident-ml-classifier:latest

docker-run-thresh:
	docker run -p 8001:8000 -e INCIDENT_THRESHOLD=0.25 incident-ml-classifier:latest

clean:
	rm -rf data/processed/*
	rm -rf src/models/artifacts/*
