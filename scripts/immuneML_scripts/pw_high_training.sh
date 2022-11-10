#!/bin/bash

#rf
immune-ml ../../immuneML/yaml_specifications/pw_high_rf.yaml ../../immuneML/pw_high_rf_out 2>&1 | tee ../../immuneML/logs/pw_high_rf.log

#knn
immune-ml ../../immuneML/yaml_specifications/pw_high_knn.yaml ../../immuneML/pw_high_knn_out 2>&1 | tee ../../immuneML/logs/pw_high_knn.log

#svm
immune-ml ../../immuneML/yaml_specifications/pw_high_svm.yaml ../../immuneML/pw_high_svm_out 2>&1 | tee ../../immuneML/logs/pw_high_svm.log