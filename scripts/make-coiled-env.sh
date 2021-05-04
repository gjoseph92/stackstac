#!/bin/sh

# cat <<EOF > requirements.txt
# dask[complete]
# IPython
# jupyter-server-proxy
# EOF
poetry export --without-hashes -E binder > requirements.txt
# echo "stackstac==$(poetry version -s)" >> requirements.txt
echo "git+https://github.com/gjoseph92/stackstac.git@1b838faf4b346538d553f105381b245116952b61#egg=stackstac[viz]" >> requirements.txt
poetry run coiled env create -n stackstac --pip requirements.txt

rm requirements.txt