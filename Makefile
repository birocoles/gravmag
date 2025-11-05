# Project variables
PROJECT=gravmag
TESTDIR=tests
CONDA_ENV=gravmag-dev
CONDA_CHANNEL := conda-forge
CONDA_FLAGS := -c $(CONDA_CHANNEL) --override-channels

# Automatically read minimum Python version from setup.cfg
PYTHON_VERSION := $(shell grep -E '^python_requires' setup.cfg | sed 's/.*>=//')

# Automatically extract dependencies from setup.cfg
CORE_DEPS := $(shell grep -A100 "install_requires" setup.cfg | grep -v "install_requires" | grep -v "^\[" | sed '/^$$/q' | tr -d '[:blank:]' | tr '\n' ' ')

.PHONY: help create-conda-env install install-dev test report style clean uninstall

help:
	@echo "Commands:"
	@echo "  create-conda-env : Create the Conda environment (prints activation instructions)"
	@echo "  install          : Install package in editable mode"
	@echo "  install-dev      : Install package, system deps, and dev tools via pip"
	@echo "  test             : Run tests with coverage (inside Conda env)"
	@echo "  report           : Open HTML coverage report"
	@echo "  style            : Format code using black (inside Conda env)"
	@echo "  clean            : Remove build and temporary files"
	@echo "  clean-env        : Remove build and temporary files (inside Conda env)"
	@echo "  uninstall        : Uninstall package"

create-conda-env:
	@if ! conda env list | grep -q $(CONDA_ENV); then \
	    echo "Creating Conda environment '$(CONDA_ENV)' with Python $(PYTHON_VERSION)..."; \
	    conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) $(CONDA_FLAGS) -y; \
	else \
	    echo "Conda environment '$(CONDA_ENV)' already exists."; \
	fi
	@echo "To activate the environment, run: conda activate $(CONDA_ENV)"

install:
	@echo "Installing dependencies from setup.cfg via conda-forge..."
	@echo "Dependencies: $(CORE_DEPS)"
	conda install -c $(CONDA_CHANNEL) $(CORE_DEPS) -y
	@echo "Installing project in editable mode..."
	pip install -e .

install-dev:
	@echo "Installing gravmag and dependencies (including dev dependencies)..."
	@if ! conda env list | grep -q $(CONDA_ENV); then \
	    make create-conda-env; \
	fi
	@echo "Installing dependencies from setup.cfg via conda-forge..."
	@echo "Dependencies: $(CORE_DEPS)"
	conda install -n $(CONDA_ENV) -c $(CONDA_CHANNEL) $(CORE_DEPS) -y
	@echo "Installing Python dependencies from setup.cfg (core + dev) in editable mode..."
	conda run -n $(CONDA_ENV) pip install -e ".[dev]"
	@echo "Installation complete. Commands 'make test' and 'make style' automatically run inside the Conda environment."

test:
	@echo "Running tests inside Conda environment '$(CONDA_ENV)'..."

	# Ensure Conda env exists
	@if ! conda env list | grep -q $(CONDA_ENV); then \
	    echo "Conda environment '$(CONDA_ENV)' does not exist. Creating it..."; \
	    make create-conda-env; \
	fi

	# Clean previous artifacts
	conda run -n $(CONDA_ENV) make clean-env

	# Install the package in editable mode
	conda run -n $(CONDA_ENV) pip install -e .

	# Run pytest with coverage, HTML report in tests/htmlcov
	conda run -n $(CONDA_ENV) pytest \
		--cov=$(PROJECT) \
		--cov-branch \
		--cov-report html:$(TESTDIR)/htmlcov \
		$(TESTDIR)

	@echo "Tests finished. HTML coverage report is available at '$(TESTDIR)/htmlcov/index.html'."

report:
	conda run -n $(CONDA_ENV) python -m webbrowser $(TESTDIR)/htmlcov/index.html

style:
	conda run -n $(CONDA_ENV) black --line-length 80 --verbose .

clean:
	@echo "Cleaning build artifacts and coverage data..."
	find . -name "*.pyc" -exec rm -v {} \; || true
	find . -name ".pytest_cache" -exec rm -rvf {} \; || true
	find . -name "__pycache__" -exec rm -rvf {} \; || true
	find . -name "*.egg-info" -exec rm -rvf {} \; || true
	find . -name "htmlcov" -exec rm -rvf {} \; || true

clean-env:
	@echo "Cleaning build artifacts and coverage data inside Conda env..."
	conda run -n $(CONDA_ENV) find . -name "*.pyc" -exec rm -v {} \; || true
	conda run -n $(CONDA_ENV) find . -name ".coverage" -exec rm -v {} \; || true
	conda run -n $(CONDA_ENV) find . -name ".pytest_cache" -exec rm -rvf {} \; || true
	conda run -n $(CONDA_ENV) find . -name "__pycache__" -exec rm -rvf {} \; || true
	conda run -n $(CONDA_ENV) find . -name "htmlcov" -exec rm -rvf {} \; || true

uninstall:
	@echo "Uninstalling Python package $(PROJECT)..."
	python -m pip uninstall -y $(PROJECT)
	@if conda env list | grep -q $(CONDA_ENV); then \
	    echo "Removing Conda environment '$(CONDA_ENV)'..."; \
	    conda env remove -n $(CONDA_ENV) -y; \
	else \
	    echo "Conda environment '$(CONDA_ENV)' does not exist."; \
	fi
