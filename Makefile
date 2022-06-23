# install, test, and clean
PROJECT=gravmag
TESTDIR=$(PROJECT)/tests

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  report    open the html test report"
	@echo "  clean     clean up build and generated files"
	@echo "  style     automatically format code"
	@echo "  uninstall   uninstall and remove from "

install:
	# Install the python package
	pip install --no-deps -e .

test:
	# Run tests using coverage and pytest
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); coverage run -m pytest; coverage html

report:
	# Show test report produced by coverage
	firefox $(TESTDIR)/htmlcov/index.html

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage" -exec rm -v {} \;
	find . -name ".pytest_cache" -exec rm -rvf {} \;
	find . -name "__pycache__" -exec rm -rvf {} \;

style:
	python -m black --line-length 80 --verbose .

uninstall:
	python -m pip uninstall $(PROJECT)
	rm -rvf *.egg-info
