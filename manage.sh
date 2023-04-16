#!/bin/bash

COMMAND=$1

echo "Requested command $COMMAND"

if [[ $COMMAND == "update_env" ]]; then
    echo "Running update_env"
    conda env update --file environment.yml --prune
    echo "Installing local NegativeClassOptimization in [--editable] mode"
    conda activate nco
    pip install -e src/NegativeClassOptimization
    pip install --force-reinstall --no-deps git+https://github.com/uio-bmi/immuneML.git
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
elif [[ $COMMAND == "run_12_all" ]]; then
    echo "Running all scripts 12."
    python scripts/script_12a_train_SN10_clean.py 2>&1 | tee data/logs/script_12a_train_SN10_clean.log
    python scripts/script_12b_train_SN10_clean_1v2.py 2>&1 | tee data/logs/script_12b_train_SN10_clean_1v2.log
    python scripts/script_12c_train_SN10_clean_1v9.py 2>&1 | tee data/logs/script_12c_train_SN10_clean_1v9.log
    python scripts/script_12d_train_SN10_clean_high_looser_95low.py 2>&1 | tee data/logs/script_12d_train_SN10_clean_high_looser_95low.log
elif [[ $COMMAND == "run_12_seed_replicates" ]]; then
    echo "run_12_seed_replicates."
    python scripts/script_12c_train_SN10_clean_1v9.py 2>&1 | tee data/logs/script_12c_train_SN10_clean_1v9.log
    python scripts/script_12d_train_SN10_clean_high_looser_95low.py 2>&1 | tee data/logs/script_12d_train_SN10_clean_high_looser_95low.log
    python scripts/script_12a_train_SN10_clean.py 2>&1 | tee data/logs/script_12a_train_SN10_clean.log
else
    echo "Command $COMMAND not detected."
fi
