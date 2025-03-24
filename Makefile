.PHONY: clean build install test lint format

# Default target
all: clean build install

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Build the package
build:
	python -m pip install --upgrade pip
	python -m pip install --upgrade build
	python -m build

# Install the package in development mode
install:
	pip install -e .

# Run tests with unittest
test:
	python -m unittest discover -s tests

# Run linting with flake8 (if available)
lint:
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src tests examples; \
	else \
		echo "flake8 not installed, skipping linting"; \
	fi

# Format code with black (if available)
format:
	@if command -v black >/dev/null 2>&1; then \
		black src tests examples; \
	else \
		echo "black not installed, skipping formatting"; \
	fi

# Build and install the wheel
wheel: clean build
	pip install dist/*.whl

# Create a virtual environment
venv:
	python -m venv .venv
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"

# Install development dependencies
dev-install: venv
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pip install -e . 