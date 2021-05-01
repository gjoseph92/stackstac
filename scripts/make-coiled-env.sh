#!/bin/sh

cat <<EOF > requirements.txt
dask[complete]
IPython
jupyter-server-proxy
EOF
echo "stackstac[binder]==$(poetry version -s)" >> requirements.txt

poetry run coiled env create -n stackstac --pip requirements.txt

rm requirements.txt