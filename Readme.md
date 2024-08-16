# Training data composition determines machine learning generalization and biological rule discovery

![version badge](https://img.shields.io/badge/version-0.9-green)

[Preprint](https://www.biorxiv.org/content/10.1101/2024.06.17.599333v2.abstract)

# TODO
- ADD DEMO TEST (with RUN TIME)
- RUN TIME General
- Systems on which was tested (LINUX)
- ADD GNU AFFERO V3 LICENSE
- Remove unnecessary files / comments in code.

- generating the right attributions names throughout the notebooks and scripts, look into it.

- Add Data without attributions, but with the precomputed files for generating everything (attributions weight a lot!)

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
# Build main dataset components, based on Absolut, called collectively MiniAbsolut 
python scripts/script_01_build_datasets.py miniabsolut 15000 5000
# For testing we recommend: python scripts/script_01_build_datasets.py miniabsolut 50 10

# Process the experimental dataset
python scripts/script_01c_build_experimental.py

# Process the per-epitope datasets
python scripts/script_01d_build_epitope_datasets.py
```

TODO: ADD TEST/DEMO Instructions.
Note: the experimental datasets and the epitope-based datasets are build using notebooks ... TODO 012c -> TODO: test on server if works now, just the dataset generation (running can test later).

2. Train the various models on the various datasets. SN10 and Logistic Regression models are trained. Datasets: Absolut data, experimental data, shuffled controls, single epitope-only data, TODO. Results are saved at TODO (Frozen_...).

TODO: RERUN ON Server the generation of train/test data for ML (no uid)
- reevaluate if needed, current code supports both with and without uid
- for logistic
- for experimental
- for per-epitope

```
# SN10: Absolut Synthetic Dataset
python scripts/script_12a_train_SN10_clean.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"
python scripts/script_12c_train_SN10_clean_1v9.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"
python scripts/script_12d_train_SN10_clean_high_looser_95low.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4"

# SN10: Shuffled Negative Controls
python scripts/script_12a_train_SN10_clean.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML_shuffled" "0,1,2,3" "0,1,2,3,4" --shuffle_labels 
python scripts/script_12c_train_SN10_clean_1v9.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML_shuffled" "0,1,2,3" "0,1,2,3,4" --shuffle_labels 
python scripts/script_12d_train_SN10_clean_high_looser_95low.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML_shuffled" "0,1,2,3" "0,1,2,3,4" --shuffle_labels 

# Logistic Regression: Absolut Synthetic Dataset
python scripts/script_12a_train_SN10_clean.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML_Linear" "0,1,2,3" "0,1,2,3,4" --logistic_regression 
python scripts/script_12c_train_SN10_clean_1v9.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML_Linear" "0,1,2,3" "0,1,2,3,4" --logistic_regression
python scripts/script_12d_train_SN10_clean_high_looser_95low.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML_Linear" "0,1,2,3" "0,1,2,3,4" --logistic_regression 

# SN10: Porebski Experimental Dataset
python scripts/script_12a_train_SN10_clean.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4" --experimental
python scripts/script_12d_train_SN10_clean_high_looser_95low.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4" --experimental

# SN10: Per-epitope datasets
python scripts/script_12a_train_SN10_clean.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4" --epitopes
python scripts/script_12c_train_SN10_clean_1v9.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4" --epitopes
python scripts/script_12d_train_SN10_clean_high_looser_95low.py "{ANALYSIS_NAME}" "data/Frozen_MiniAbsolut_ML" "0,1,2,3" "0,1,2,3,4" --epitopes

```

3. Compute the ID and OOD performances.

```
# ID, OOD: Absolut Synthetic Dataset
python scripts/script_14_frozen_transfer_performance.py 1 0 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance.tsv"
python scripts/script_14_frozen_transfer_performance.py 0 1 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance.tsv"

# ID, OOD: Shuffled Negative Controls
python scripts/script_14_frozen_transfer_performance.py 1 0 "data/Frozen_MiniAbsolut_ML_shuffled" "data/Frozen_MiniAbsolut_ML_shuffled/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML_shuffled/openset_performance.tsv"
python scripts/script_14_frozen_transfer_performance.py 0 1 "data/Frozen_MiniAbsolut_ML_shuffled" "data/Frozen_MiniAbsolut_ML_shuffled/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML_shuffled/openset_performance.tsv"

# ID, OOD: Logistic Regression
python scripts/script_14_frozen_transfer_performance.py 1 0 "data/Frozen_MiniAbsolut_ML_Linear" "data/Frozen_MiniAbsolut_ML_Linear/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML_Linear/openset_performance.tsv"
python scripts/script_14_frozen_transfer_performance.py 0 1 "data/Frozen_MiniAbsolut_ML_Linear" "data/Frozen_MiniAbsolut_ML_Linear/closed_performance.tsv" "data/Frozen_MiniAbsolut_ML_Linear/openset_performance.tsv"

# ID, OOD: Experimental Dataset
python scripts/script_14_frozen_transfer_performance.py 1 0 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance_experimental_data.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance_experimental_data.tsv"
python scripts/script_14_frozen_transfer_performance.py 0 1 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance_experimental_data.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance_experimental_data.tsv"

# ID, OOD: Per-epitope datasets
python scripts/script_14b_frozen_transfer_performance.py 1 0 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance_epitopes_pos.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance_epitopes_pos.tsv"
python scripts/script_14b_frozen_transfer_performance.py 0 1 "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML/closed_performance_epitopes_pos.tsv" "data/Frozen_MiniAbsolut_ML/openset_performance_epitopes_pos.tsv"
python scripts/script_14c_epi_frozen_transfer_jsd.py
```

4. Run the interpretability pipeline.

Get [Absolut!](https://github.com/csi-greifflab/Absolut): clone the library, install AbsolutNoLib (according to the instructions), and save the path to the `AbsolutNoLib` in the environment variable `ABSOLUTNOLIB_PATH`.

```
export ABSOLUTNOLIB_PATH=$PWD/AbsolutNoLib

# Compute the energy contributions using Absolut. Note: to restrict computational costs the computation is done only for the main MiniAbsolut train/test split and one of the seeds. 
python scripts/script_16_compute_energy_contributions.py
```

Note: if the latest Absolut! doesn't work below, reset to an earlier commit that was used in this work by running: `git reset --hard 0f672a19c9fdec561e4d6d2470471ea016f949ad`.

Next: compute the attributions.

```
# Interpretability: SN10
python scripts/script_15_compute_attributions.py "Test" "data/Frozen_MiniAbsolut_ML"

# Interpretability: Shuffled negative control
python scripts/script_15_compute_attributions.py "Test" "data/Frozen_MiniAbsolut_ML_shuffled"

# Interpretability: Logistic Regression: No need to run DeepLift, it's based on the model weights (done in Analyses).
python scripts/script_01_build_datasets.py adapt_attributions_for_linear

# Interpretability: Experimental Dataset
python scripts/script_15_compute_attributions.py "Test" "data/Frozen_MiniAbsolut_ML" --experimental

# Interpretability: Per-epitope datasets
python scripts/script_15_compute_attributions.py "Test" "data/Frozen_MiniAbsolut_ML" --epitopes_only "PositiveSet_Epitope"
```

## Analyse

Once we have the precomputations (either downloaded or precomputed from scratch using the steps from above), we perform the analyses using notebooks.

Section 1: Training dataset sequence composition influences prediction performance in ID and OOD binary classification tasks
- ID and OOD on synthetic data and sequence similarity: `0a1_Section_1.ipynb`.
- ID and OOD on experimental data: `0a3_Section_1_experimental.ipynb`.

Section 2: Training dataset composition determines the accuracy of biological rule recovery
- Correlations between ground truth energy per sequence and per amino acid and logits / attributions: `0b_Section_2.ipynb`.
- Correlation between binding strength and logit in the experimental dataset: `0b3_Section_2_experimental.ipynb`.

Supplementary Materials
- Negative control: shuffled positive and negative in training sets labels: `0a2_Section_1_shuffled.ipynb`, `0b2_Section_2_shuffled.ipynb`
- Logistic models: `0a4_Section_1_logistic.ipynb`, `0b4_Section_2_logistic.ipynb`

Supplementary Text 1: Evaluation of the impact of sequence and label similarity between train and test on prediction accuracy 
- TODO

Supplementary Text 2: Epitope-based analysis: ID, OOD, rule discovery
- `0s2_Epitope-based_Section_1_and_2.ipynb`

