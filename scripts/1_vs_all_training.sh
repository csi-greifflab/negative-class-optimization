#!/bin/bash

#rf
immune-ml ../immuneML/yaml_specifications/1_vs_all_rf.yaml ../immuneML/1_vs_all_rf_out 2>&1 | tee ../immuneML/logs/1_vs_all_rf.log

#knn
immune-ml ../immuneML/yaml_specifications/1_vs_all_knn.yaml ../immuneML/1_vs_all_knn_out 2>&1 | tee ../immuneML/logs/1_vs_all_knn.log

#svm
immune-ml ../immuneML/yaml_specifications/1_vs_all_svm.yaml ../immuneML/1_vs_all_svm_out 2>&1 | tee ../immuneML/logs/1_vs_all_svm.log