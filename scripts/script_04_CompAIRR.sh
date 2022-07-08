#!/bin/bash

## Test overlap analysis
# TEST_DATA_PATH="../data/CompAIRR/test"
# docker run -it -v ${PWD}/${TEST_DATA_PATH}:/opt/compairr torognes/compairr \
#     --ignore-genes --ignore-counts \
#     --matrix \
#     --threads 12 \
#     --output test_output.tsv \
#     --pairs test_pairs.tsv \
#     AG1_AIRR.tsv \
#     AG2_AIRR.tsv

run_compairr_overlaps() {
    DIR=${PWD}/${COMPAIRR_DATA_PATH}
    docker run -it -v $DIR:/opt/compairr torognes/compairr \
        --ignore-genes --ignore-counts \
        --threads 12 \
        --matrix \
        --differences ${DIFFERENCES} \
        --log d${DIFFERENCES}.log \
        --output d${DIFFERENCES}_output.tsv \
        --pairs d${DIFFERENCES}_pairs.tsv \
        ${COMPAIRR_FILE_1} \
        ${COMPAIRR_FILE_2}
}

COMPAIRR_DATA_PATH="data/CompAIRR"
COMPAIRR_FILE_1="AIRR_1.tsv"
COMPAIRR_FILE_2="AIRR_2.tsv"

# Main
docker pull torognes/compairr

DIFFERENCES=0
run_compairr_overlaps

DIFFERENCES=1
run_compairr_overlaps

DIFFERENCES=2
run_compairr_overlaps
