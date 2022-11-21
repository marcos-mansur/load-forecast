prepare:
	@black .
	@isort .
	@mypy src
	@pylint src
	@flake8 src
	@echo Good to Go!

check:
	@black . --check
	@isort . --check
	@mypy src
	@flake8 src
	@pylint src
	@echo Good to Go!

docs:
	@mkdocs build --clean

docs-serve:
	@mkdocs serve

test:
	@pytest --cov src

test-cov:
	@pytest --cov src --cov-report xml:coverage.xml
.PHONY: docs