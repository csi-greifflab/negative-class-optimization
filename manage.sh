#!/bin/bash

COMMAND=$1

echo "Requested command $COMMAND"

if [[ $COMMAND == "update_env" ]]; then
    echo "Running update_env"
    conda env update --file environment.yml --prune
    echo "Installing local NegativeClassOptimization in [--editable] mode"
    conda activate ab-negative-training
    pip install -e src/NegativeClassOptimization
    pip install --force-reinstall --no-deps git+https://github.com/uio-bmi/immuneML.git
    # echo "Installing local immuneML in [--editable] mode"
    # conda activate ab-negative-training
    # pip install -e immuneML/immuneML
elif [[ $COMMAND == "install_env" ]]; then
    echo "Running install_env"
    conda env create --force --file environment.yml
else
    echo "Command $COMMAND not detected."
fi
