# test and clean
TESTDIR=GravMag/tests/tmp-test-dir-with-unique-name

help:
	@echo "Commands:"
	@echo ""
	@echo "  test      run the test suite and generate html report"
	@echo "  report    open the html test report"
	@echo "  clean     clean up generated files"
	@echo ""

test:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	# Run tests using coverage and pytest
	cd $(TESTDIR)
	coverage run -m pytest
	coverage html

report:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	# Show test report produced by coverage
	cd $(TESTDIR)
	firefox htmlcov/index.html

clean:
	rm -rvf __pycache__
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf __pycache__ .coverage .cache .pytest_cache
	rm -rvf $(TESTDIR)
