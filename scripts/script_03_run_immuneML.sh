#!/bin/bash

## pairwise_analysis_1
immune-ml ../immuneML/pairwise_analysis_1.yml \
    ../immuneML/pairwise_analysis_1_out 2>&1 \
    | tee ../immuneML/pairwise_analysis_1.log

## 1_vs_all_analysis_1
# immune-ml ../immuneML/1_vs_all_analysis_1.yml \
#     ../immuneML/1_vs_all_analysis_1_out 2>&1 \
#     | tee ../immuneML/1_vs_all_analysis.log