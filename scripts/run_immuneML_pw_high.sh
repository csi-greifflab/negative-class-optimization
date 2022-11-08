#!/bin/bash

#svm
python build_immuneML_yaml.py pw_high oh svm accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/pw_high_svm.yaml

immune-ml ../immuneML/yaml_specifications/pw_high_svm.yaml ../immuneML/pw_high_svm_out 2>&1 | tee ../immuneML/logs/pw_high_svm.log

#rf
python build_immuneML_yaml.py pw_high oh rf accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/pw_high_rf.yaml


immune-ml ../immuneML/yaml_specifications/pw_high_rf.yaml ../immuneML/pw_high_rf_out 2>&1 | tee ../immuneML/logs/pw_high_rf.log

#knn
python build_immuneML_yaml.py pw_high oh knn accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/pw_high_knn.yaml

immune-ml ../immuneML/yaml_specifications/pw_high_knn.yaml ../immuneML/pw_high_knn_out 2>&1 | tee ../immuneML/logs/pw_high_knn.log