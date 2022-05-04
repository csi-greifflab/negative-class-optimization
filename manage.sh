#!/bin/bash

COMMAND=$1

echo "Requested command $COMMAND"

if [[ $COMMAND == "update_env" ]]; then
    echo "Running update_env"
    conda env create --force --file environment.yml
else
    echo "Command $COMMAND not detected."
fi
