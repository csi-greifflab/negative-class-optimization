#!/bin/bash

COMMAND=$1

echo "Requested command $COMMAND"

if [[ $COMMAND == "update_env" ]]; then
    echo "Running update_env"
    conda env update --file environment.yml --prune
    echo "Installing local NegativeClassOptimization in [--editable] mode"
    conda activate nco
    pip install -e src/NegativeClassOptimization
    # echo "Installing local immuneML in [--editable] mode"
    # conda activate ab-negative-training
    # pip install -e immuneML/immuneML
elif [[ $COMMAND == "install_env" ]]; then
    echo "Running install_env"
    conda env create --force --file environment.yml
elif [[ $COMMAND == "get_700k" ]]; then
    echo "Fetching the 700k dataset with DVC."
    mkdir -p data/globals
    dvc get --out data/globals/slack_1_global.tsv . data/globals/slack_1_global.tsv
elif [[ $COMMAND == "update_dvc_pipelines" ]]; then
    echo "Updating and reproducing dvc pipelines."
    dvc repro
    dvc commit
else
    echo "Command $COMMAND not detected."
fi
