LINT_PATHS=mario_brain/ tests/

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

commit-checks: format lint

test:
	@echo "Running tests..."
	PYTHONPATH=. pytest tests/ --tb=short

coverage:
	@echo "Running tests with coverage..."
	PYTHONPATH=. pytest tests/ --tb=short --cov=spark_exec_predictor --cov-report=html:coverage_html_report --cov-report=term-missing
	@echo "Coverage report generated in coverage_html_report/index.html"