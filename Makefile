PYTHON = .mlVenv\Scripts\python.exe
PIP    = .mlVenv\Scripts\pip.exe

.PHONY: install data clean features train evaluate app test lint

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) setup.py develop

data:
	$(PYTHON) src/data/make_dataset.py

clean:
	$(PYTHON) src/data/clean_data.py

features:
	$(PYTHON) src/data/build_features.py

train:
	$(PYTHON) src/models/train_model.py

evaluate:
	$(PYTHON) src/models/evaluate_model.py

app:
	$(PYTHON) app/app.py

backend:
	$(PYTHON) -m uvicorn app.backend.main:app --reload --port 8000

frontend:
	$(PYTHON) app/frontend/dashboard.py

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m flake8 src/ app/ tests/ --max-line-length=120

all: install data clean features train evaluate
