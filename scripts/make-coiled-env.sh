#!/bin/sh

cat <<EOF > requirements.txt
bokeh>=0.13.0
IPython
jupyter-server-proxy
EOF
poetry export --dev --without-hashes | grep -E 'dask|distributed' >> requirements.txt
echo "stackstac[viz]==$(poetry version -s)" >> requirements.txt

poetry run coiled env create -n stackstac --pip requirements.txt

rm requirements.txt