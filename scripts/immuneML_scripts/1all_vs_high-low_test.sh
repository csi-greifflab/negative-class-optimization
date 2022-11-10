#!/bin/bash

#svm
immune-ml ../../immuneML/yaml_specifications/1all_vs_high-low_svm_test.yaml ../../immuneML/1all_vs_high-low_svm_test 2>&1 | tee ../../immuneML/logs/1all_vs_high-low_svm_test.log

#knn
immune-ml ../../immuneML/yaml_specifications/1all_vs_high-low_knn_test.yaml ../../immuneML/1all_vs_high-low_knn_test 2>&1 | tee ../../immuneML/logs/1all_vs_high-low_knn_test.log

#rf
immune-ml ../../immuneML/yaml_specifications/1all_vs_high-low_rf_test.yaml ../../immuneML/1all_vs_high-low_rf_test 2>&1 | tee ../../immuneML/logs/1all_vs_high-low_rf_test.log