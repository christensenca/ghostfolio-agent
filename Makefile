.PHONY: install test lint build publish clean eval eval-setup eval-dry-run

# Install package in editable mode with all extras
install:
	pip install -e ".[dev,server]"

# Run unit tests
test:
	pytest tests/ -v

# Build wheel and sdist
build:
	python -m build

# Publish to PyPI
publish:
	twine upload dist/*

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info/ src/*.egg-info/

# Eval commands (require Ghostfolio instance)
eval:
	python -m evals.eval_runner --setup

eval-setup:
	python -m scripts.setup_eval

eval-dry-run:
	python -m evals.eval_runner --dry-run
