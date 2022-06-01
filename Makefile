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
	coverage run -m pytest
	coverage html

report:
	# Show test report produced by coverage
	firefox htmlcov/index.html

clean:
	rm -rvf __pycache__
