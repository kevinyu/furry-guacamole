#!/usr/bin/env bash

: "${CONDA_PREFIX?Need to be in conda env! Create (conda create --env [name]) or activate (source activate [name])}"

if [ ! -d "$CONDA_PREFIX" ]; then
   echo "Conda env at $CONDA_PREFIX does not exist"
   exit 1
fi

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"


ACTIVATE_ENV_VARS="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
DEACTIVATE_ENV_VARS="$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
touch $ACTIVATE_ENV_VARS
touch $DEACTIVATE_ENV_VARS

echo "Setting up env variable scripts in $ACTIVATE_ENV_VARS and $DEACTIVATE_ENV_VARS"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "export PROJECT_ROOT=${PROJECT_ROOT}" > $ACTIVATE_ENV_VARS
echo "export PYTHONPATH=\"\${PROJECT_ROOT}/src:\${PROJECT_ROOT}/lib\"" >> $ACTIVATE_ENV_VARS

echo "unset PROJECT_ROOT" > $DEACTIVATE_ENV_VARS
echo "unset PYTHONPATH" >> $DEACTIVATE_ENV_VARS

