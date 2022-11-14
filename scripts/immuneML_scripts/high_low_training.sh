#!/bin/bash

#rf
immune-ml ../../immuneML/yaml_specifications/high_low_rf.yaml ../../immuneML/high_low_rf_out 2>&1 | tee ../../immuneML/logs/high_low_rf.log

#knn
immune-ml ../../immuneML/yaml_specifications/high_low_knn.yaml ../../immuneML/high_low_knn_out 2>&1 | tee ../../immuneML/logs/high_low_knn.log

#svm
immune-ml ../../immuneML/yaml_specifications/high_low_svm.yaml ../../immuneML/high_low_svm_out 2>&1 | tee ../../immuneML/logs/high_low_svm.log