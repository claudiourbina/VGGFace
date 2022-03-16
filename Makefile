.PHONY: test
test:
	@echo "makefile"
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=vggface tests