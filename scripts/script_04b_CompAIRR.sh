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
        --log results/overlaps_d${DIFFERENCES}.log \
        --output results/overlaps_d${DIFFERENCES}_output.tsv \
        --pairs results/overlaps_d${DIFFERENCES}_pairs.tsv \
        ${COMPAIRR_FILE} ${COMPAIRR_FILE}
}

run_compairr_clustering_per_antigen() {
    DIR=${PWD}/${COMPAIRR_DATA_PATH}
    docker run -it -v $DIR:/opt/compairr torognes/compairr \
        --ignore-genes --ignore-counts \
        --threads 12 \
        --cluster \
        --differences ${DIFFERENCES} \
        --log results/clustering_per_antigen_d${DIFFERENCES}.log \
        --output results/clustering_per_antigen_d${DIFFERENCES}_output.tsv \
        ${COMPAIRR_FILE}
}

run_compairr_clustering_all_antigens() {
    DIR=${PWD}/${COMPAIRR_DATA_PATH}
    docker run -it -v $DIR:/opt/compairr torognes/compairr \
        --ignore-genes --ignore-counts \
        --threads 12 \
        --cluster \
        --differences ${DIFFERENCES} \
        --log results/clustering_d${DIFFERENCES}.log \
        --output results/clustering_d${DIFFERENCES}_output.tsv \
        ${COMPAIRR_GLOBAL_REPERTOIRE}
}

COMPAIRR_DATA_PATH="data/CompAIRR"
COMPAIRR_FILE="prepared_data/AIRR.tsv"
COMPAIRR_GLOBAL_REPERTOIRE="prepared_data/AIRR_global_repertoire.tsv"

# Main
docker pull torognes/compairr

mkdir ${COMPAIRR_DATA_PATH}/results

DIFFERENCES=0
run_compairr_overlaps
run_compairr_clustering_per_antigen
run_compairr_clustering_all_antigens

DIFFERENCES=1
run_compairr_overlaps
run_compairr_clustering_per_antigen
run_compairr_clustering_all_antigens

DIFFERENCES=2
run_compairr_overlaps
run_compairr_clustering_per_antigen
run_compairr_clustering_all_antigens
