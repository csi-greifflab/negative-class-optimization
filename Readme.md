# Training data composition determines machine learning generalization and biological rule discovery

![version badge](https://img.shields.io/badge/version-0.9-green)

[Preprint](https://www.biorxiv.org/content/10.1101/2024.06.17.599333v2.abstract)

## Setup

### A. Environment
Once you've cloned the repository, install the environment using conda based on `environment.yml` file and install the local package.
```
conda env create --force --file environment.yml
conda activate nco
pip install -e src/NegativeClassOptimization
```

### B. Data

### Absolut

TODO: used the custom csv. # TODO add the link from GDrive or save in Github!

Download `Absolut` data. Get the doi csv from [data source](https://archive.norstore.no/pages/public/toc.jsf?code=A3TL2NWADL), save it to data/Absolut/toc_doi10.11582_2021.00063.csv. Then run:

```
python scripts/script_01_build_datasets.py download_absolut
python scripts/script_01_build_datasets.py unzip_rawbindingsmurine  # TODO: not absolutely necessary, as we only used the *PerClass folder
```

### Experimental data

TODO


## Precomputations

TODO: We provide the precomputed results at ...
TODO: clean uids from test_dataset in original data. resolve when multiple.

Alternatively, if you want to run the precomputations, follow the further instructions.

1. Generate the Miniabsolut dataset(s). Each MiniAbsolut dataset is a subset of the Absolut dataset with a specific number of samples per class for easier further assembly of training and test datasets, having already splitted training and test data, and removing duplicates and possible intersections. The script generates the datasets and saves them to `data/MiniAbsolut` and `data/MiniAbsolut_Splits`. Run:

```
python scripts/script_01_build_datasets.py miniabsolut 15000 5000
TEST: python scripts/script_01_build_datasets.py miniabsolut 50 10
```

1. Train the various models on the various datasets. SN10 and Logistic Regression models are trained. Datasets: Absolut data, experimental data, shuffled controls, single epitope-only data, TODO. Results are saved at TODO (Frozen_...).


```
# SN10: Absolut
python scripts/script_12a_train_SN10_clean.py "Test" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"
python scripts/script_12c_train_SN10_clean_1v9.py "Test" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"
python scripts/script_12d_train_SN10_clean_high_looser_95low.py "Test" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"


# SN10: Experimental: TODO
script_12a_train_SN10_clean.py
script_12c_train_SN10_clean_1v9.py
script_12d_train_SN10_clean_high_looser_95low.py

# SN10: Shuffled controls
# TODO

# Logistic Regression: Absolut
# TODO

# TODO

```

3. Compute the ID and OOD performances.

```
# Compute ID
python scripts/script_14_frozen_transfer_performance.py 1 0 "data/Frozen_MiniAbsolut_ML" "data/closed_performance.tsv" "data/openset_performance.tsv"
# Compute OOD
python scripts/script_14_frozen_transfer_performance.py 0 1 "data/Frozen_MiniAbsolut_ML" "data/closed_performance.tsv" "data/openset_performance.tsv"
```

4. Run the interpretability pipeline.

```
# Compute the energy contributions using Absolut
script_16_compute_energy_contributions.py

# Run the interpretability pipeline
script_15_compute_attributions.py
```

## Analysis

Once we have almost everything precomputed, we perform the analyses using notebooks.

1. TODO
