#!/bin/bash

#svm
immune-ml ../../immuneML/yaml_specifications/pw_high_cross_svm_test.yaml ../../immuneML/pw_high_cross_svm_test 2>&1 | tee ../../immuneML/logs/pw_high_cross_svm_test.log

#knn
immune-ml ../../immuneML/yaml_specifications/pw_high_cross_knn_test.yaml ../../immuneML/pw_high_cross_knn_test 2>&1 | tee ../../immuneML/logs/pw_high_cross_knn_test.log

#rf
immune-ml ../../immuneML/yaml_specifications/pw_high_cross_rf_test.yaml ../../immuneML/pw_high_cross_rf_test 2>&1 | tee ../../immuneML/logs/pw_high_cross_rf_test.log