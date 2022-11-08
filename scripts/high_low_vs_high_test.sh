#!/bin/bash

#svm
immune-ml ../immuneML/yaml_specifications/high_low_vs_high_svm_test.yaml ../immuneML/high_low_vs_high_svm_test 2>&1 | tee ../immuneML/logs/high_low_vs_high_svm_test.log

#knn
immune-ml ../immuneML/yaml_specifications/high_low_vs_high_knn_test.yaml ../immuneML/high_low_vs_high_knn_test 2>&1 | tee ../immuneML/logs/high_low_vs_high_knn_test.log

#rf
immune-ml ../immuneML/yaml_specifications/high_low_vs_high_rf_test.yaml ../immuneML/high_low_vs_high_rf_test 2>&1 | tee ../immuneML/logs/high_low_vs_high_rf_test.log