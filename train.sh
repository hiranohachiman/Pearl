#!/bin/bash

yecho(){ # yellow echo
    echo "\e[1;33m$1\e[m"
}

arg=$1 
if [ "$arg" = "pdb" ]; then
    yecho 'Launch pdb ...'
    PYTHON_CMD='poetry run python -m pdb -c c'
else 
    yecho 'Launch without pdb ...'
    PYTHON_CMD='poetry run python'
fi

$PYTHON_CMD pearl/cli.py train -f configs/xlmr/base/da-estimator-base.yaml && \
sh validate.sh
