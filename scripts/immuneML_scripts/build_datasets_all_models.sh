#!/bin/bash

#rf
python3 build_immuneML_yaml.py high_low oh rf accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/high_low_rf.yaml

python3 build_immuneML_yaml.py pw_high oh rf accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/pw_high_rf.yaml

python3 build_immuneML_yaml.py 1_vs_all oh rf accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/1_vs_all_rf.yaml

#knn
python3 build_immuneML_yaml.py high_low oh knn accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/high_low_knn.yaml

python3 build_immuneML_yaml.py pw_high oh knn accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/pw_high_knn.yaml

python3 build_immuneML_yaml.py 1_vs_all oh knn accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/1_vs_all_knn.yaml

#svm
python3 build_immuneML_yaml.py high_low oh svm accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/high_low_svm.yaml

python3 build_immuneML_yaml.py pw_high oh svm accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/pw_high_svm.yaml

python3 build_immuneML_yaml.py 1_vs_all oh svm accuracy --refit_best --metadata -o ~/negative-class-optimization/immuneML/yaml_specifications/1_vs_all_svm.yaml
