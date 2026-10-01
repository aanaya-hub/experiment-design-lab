# Makefile — the four commands this project needs, with no arguments to remember.
#
# WHY A MAKEFILE: a README that says "run these commands in this order" is a promise the repository
# cannot keep. A Makefile is that promise as executable code, and `make check` fails loudly if the
# analysis, the tests or the notebook have drifted apart.
#
# PYTHON is overridable so the project works with or without a virtual environment:
#   make check PYTHON=./venv-edl/bin/python

PYTHON ?= python3

.PHONY: help install analyse test probe notebook check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

analyse:  ## Run the full analysis and write reports/results.json
	$(PYTHON) src/main.py

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

probe:  ## Read-only feasibility probe over the sibling project's data
	$(PYTHON) scripts/explore_markdown.py

notebook:  ## Rebuild and execute the notebook from source
	$(PYTHON) scripts/build_notebook.py

check: analyse test notebook  ## The build gate: analysis, tests and notebook must all pass
	@echo "OK — analysis, tests and notebook all completed"

clean:  ## Remove caches and generated reports
	rm -rf .pytest_cache **/__pycache__ __pycache__ src/__pycache__ tests/__pycache__ \
	  scripts/__pycache__
	rm -f reports/results.json
