PYTHONPATH := services/clip-factory/src

.PHONY: test api worker up down

test:
	PYTHONPATH=$(PYTHONPATH) python3 -m unittest discover -s services/clip-factory/tests -v

api:
	PYTHONPATH=$(PYTHONPATH) python3 -m uvicorn clip_factory.api:app --host 0.0.0.0 --port 8000

worker:
	PYTHONPATH=$(PYTHONPATH) python3 -m clip_factory.worker

up:
	docker-compose up --build

down:
	docker-compose down

