# test and clean

help:
	@echo "Commands:"
	@echo ""
	@echo "  test      run the test suite and generate html report"
	@echo "  report    open the html test report"
	@echo "  clean     clean up generated files"
	@echo ""

test:
	# Run tests using coverage and pytest
	cd gravmag_code/tests; coverage run -m pytest; coverage html

report:
	# Show test report produced by coverage
	cd gravmag_code/tests; firefox htmlcov/index.html

clean:
	rm -rvf __pycache__
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf __pycache__ .coverage .cache .pytest_cache .egg-info
