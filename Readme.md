# Training data composition determines machine learning generalization and biological rule discovery

![version badge](https://img.shields.io/badge/version-0.9-green)

[Preprint](https://www.biorxiv.org/content/10.1101/2024.06.17.599333v2.abstract)

# TODO
- ADD DEMO TEST (with RUN TIME)
- RUN TIME General
- Systems on which was tested (LINUX)
- ADD GNU AFFERO V3 LICENSE

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

Download and unpack the dataset from https://zenodo.org/records/10123621.


## Precompute

TODO: We provide the precomputed results at ...
TODO: clean uids from test_dataset in original data. resolve when multiple. Clean analysis names -> "main_analysis".

Alternatively, if you want to run the precomputations, follow the further instructions.

1. Generate the Miniabsolut dataset(s). Each MiniAbsolut dataset is a subset of the Absolut dataset with a specific number of samples per class for easier further assembly of training and test datasets, having already splitted training and test data, and removing duplicates and possible intersections. The script generates the datasets and saves them to `data/MiniAbsolut` and `data/MiniAbsolut_Splits`. Run:

```
python scripts/script_01_build_datasets.py miniabsolut 15000 5000
# For testing we recommend: python scripts/script_01_build_datasets.py miniabsolut 50 10
```

2. Train the various models on the various datasets. SN10 and Logistic Regression models are trained. Datasets: Absolut data, experimental data, shuffled controls, single epitope-only data, TODO. Results are saved at TODO (Frozen_...).

```
# SN10: Absolut Synthetic Dataset
python scripts/script_12a_train_SN10_clean.py "Test" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"
python scripts/script_12c_train_SN10_clean_1v9.py "Test" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"
python scripts/script_12d_train_SN10_clean_high_looser_95low.py "Test" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"

# SN10: Porebski Experimental Dataset
# TODO

# SN10: Shuffled controls
# TODO

# Logistic Regression: Absolut
# TODO
```

3. Compute the ID and OOD performances.

```
# Compute ID
python scripts/script_14_frozen_transfer_performance.py 1 0 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance.tsv"
# Compute OOD
python scripts/script_14_frozen_transfer_performance.py 0 1 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance.tsv"
```

4. Run the interpretability pipeline.

Get [Absolut!](https://github.com/csi-greifflab/Absolut): clone the library, install AbsolutNoLib (according to the instructions), and save the path to the `AbsolutNoLib` in the environment variable `ABSOLUTNOLIB_PATH`.

```
export ABSOLUTNOLIB_PATH=$PWD/AbsolutNoLib

# Compute the energy contributions using Absolut. Note: to restrict computational costs the computation is done only for the main MiniAbsolut train/test split and seed. 
python scripts/script_16_compute_energy_contributions.py
```

Note: if the latest Absolut! doesn't work below, reset to an earlier commit that was used by us by running: `git reset --hard 0f672a19c9fdec561e4d6d2470471ea016f949ad`.

Next: compute the attributions.

```
# Run the interpretability pipeline
python scripts/script_15_compute_attributions.py "TEST" "data/Frozen_MiniAbsolut_ML"
```

## Analyse

Once we have the precomputations (either downloaded or precomputed from scratch using the steps from above), we perform the analyses using notebooks.

Prerequisites
- Frozen_ML:
  - closed/open perf
  - Attribution/LogitEnergyCorrelations.tsv (correction of error? check old nb on server TODO)
- 

Section 1: Training dataset sequence composition influences prediction performance in ID and OOD binary classification tasks
- ID and OOD on synthetic data and sequence similarity: `0a1_Section_1.ipynb`.
- ID and OOD on experimental data: TODO

Section 2: Training dataset composition determines the accuracy of biological rule recovery
- Correlations between ground truth energy per sequence and per amino acid and logits / attributions: `0b_Section_2.ipynb`
  - TODO: attributions name: remove or add "Test" or add as parameter!
- Correlation between binding strength and logit in the experimental dataset: TODO

Supplementary Materials
- Negative control: shuffled positive and negative in training sets labels
  -  .
- Logistic models
  - Correlations between ground truth energy per sequence and per amino acid and logits / attributions: 
  - Negative control (correlations with logits after shuffling the weights): 

Supplementary Text 1: Evaluation of the impact of sequence and label similarity between train and test on prediction accuracy 
- .

Supplementary Text 2: Antigen versus epitope-based analysis: ID, OOD, rule discovery
- .

