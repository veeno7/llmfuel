install:
	python -m pip install -e .[all]

test:
	pytest -q

benchmark:
	python -m fuel.cli benchmark --limit 10 --output benchmarks/results.json

download-model:
	python -m fuel.cli download-model
