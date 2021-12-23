.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

all: ml_federated_customer_journey docs

ml_federated_customer_journey: $(SRC)
	nbdev_build_lib
	touch ml_federated_customer_journey

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi conda_release
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist