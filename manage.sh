#!/bin/bash

COMMAND=$1

echo "Requested command $COMMAND"

if [[ $COMMAND == "update_env" ]]; then
    echo "Running update_env"
    # conda env create --force --file environment.yml
    conda env update --file environment.yml --prune
    # echo "Installing local immuneML in [--editable] mode"
    # pip install -e immuneML/immuneML
elif [[ $COMMAND == "install_env" ]]; then
    echo "Running install_env"
    conda env create --force --file environment.yml
else
    echo "Command $COMMAND not detected."
fi
