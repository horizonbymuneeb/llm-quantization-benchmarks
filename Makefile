PY ?= python3
CONFIG ?= config.yaml
# The package lives under src/, so the module entry points need it on the path.
RUN = PYTHONPATH=src $(PY)

.PHONY: help install test test-fast sweep inspect calibrate report clean lint check

help:
	@echo "targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  test        run the full pytest suite"
	@echo "  sweep       run the full quantisation sweep and write reports/sweep_report.json"
	@echo "  inspect     evaluate one configuration (FMT=int8 make inspect)"
	@echo "  calibrate   print calibrated activation ranges"
	@echo "  report      print the digest of the last report"
	@echo "  clean       remove caches and generated reports"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

sweep:
	$(RUN) -m quantbench.cli sweep --config $(CONFIG) --csv reports/rows.csv

inspect:
	$(RUN) -m quantbench.cli inspect --config $(CONFIG) --format $(or $(FMT),int4)

calibrate:
	$(RUN) -m quantbench.cli calibrate --config $(CONFIG)

report:
	$(RUN) -c "from quantbench.report import load_report, summarise_report; \
print(summarise_report(load_report('reports/sweep_report.json')))"

lint:
	$(PY) -m compileall -q src tests

check: lint test

clean:
	rm -rf reports .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
