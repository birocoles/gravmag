# Project variables
PROJECT=gravmag
TESTDIR=tests
CONDA_ENV=gravmag-dev

# Automatically read minimum Python version from setup.cfg
PYTHON_VERSION := $(shell grep -E '^python_requires' setup.cfg | sed 's/.*>=//')

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
	@echo "  uninstall        : Uninstall package"

create-conda-env:
	@if ! conda env list | grep -q $(CONDA_ENV); then \
	    echo "Creating Conda environment '$(CONDA_ENV)' with Python $(PYTHON_VERSION)..."; \
	    conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y; \
	else \
	    echo "Conda environment '$(CONDA_ENV)' already exists."; \
	fi
	@echo "To activate the environment, run: conda activate $(CONDA_ENV)"

install:
	pip install -e .

install-dev:
	@if ! conda env list | grep -q $(CONDA_ENV); then \
	    make create-conda-env; \
	fi
	@echo "Installing system-level dependencies via Conda (GDAL/PROJ for rasterio)..."
	conda install -n $(CONDA_ENV) -c conda-forge gdal -y
	@echo "Installing Python dependencies from setup.cfg (core + dev) in editable mode..."
	conda run -n $(CONDA_ENV) pip install -e ".[dev]"
	@echo "Installation complete. Commands 'make test' and 'make style' automatically run inside the Conda environment."

test:
	conda run -n $(CONDA_ENV) pytest --cov=$(PROJECT) $(TESTDIR) --cov-report html

report:
	python -m webbrowser $(TESTDIR)/htmlcov/index.html

style:
	conda run -n $(CONDA_ENV) black --line-length 80 --verbose .

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".pytest_cache" -exec rm -rvf {} \;
	find . -name "__pycache__" -exec rm -rvf {} \;
	find . -name "*.egg-info" -exec rm -rvf {} \;

uninstall:
	@echo "Uninstalling Python package $(PROJECT)..."
	python -m pip uninstall -y $(PROJECT)
	@if conda env list | grep -q $(CONDA_ENV); then \
	    echo "Removing Conda environment '$(CONDA_ENV)'..."; \
	    conda env remove -n $(CONDA_ENV) -y; \
	else \
	    echo "Conda environment '$(CONDA_ENV)' does not exist."; \
	fi
