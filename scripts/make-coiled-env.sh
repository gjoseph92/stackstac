#!/bin/sh

poetry export --without-hashes -E binder > requirements.txt
echo "stackstac==$(poetry version -s)" >> requirements.txt
poetry run coiled env create -n stackstac --pip requirements.txt

rm requirements.txt