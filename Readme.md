# Negative dataset optimization

We use this repository for the `NegativeDatasetOptimization` project.

Please follow this readme to setup everything.

## Setup

Once you've cloned the repository, run:

```
bash manage.sh install_env
conda activate ab-negative-training
bash manage.sh update_env  # updates and installs local packages
dvc pull  # fetches the data
```

This will setup the environment and required data. In particular, this will fetch the `data` directory, with all required data, as well as `immuneML` directory, where immuneML yaml specifications, logs, as well as output files are stored.

To add a new library dependence, please add it manually in the `environment.yml` file and run `bash manage.sh update_env`.

## Data and models

We use [DVC](https://dvc.org/doc/start/data-management) to sync data and models across the team.

### DVC Workflow

Once you've made changes to a `DVC`-tracked directory, please add the changes to `dvc` and `git` (provided example is for the `data` directory):

```
dvc add data
git commit data.dvc -m "data directory updates"
dvc push
```

Note that to control file size, some large files are ignored (check `.dvcignore`).

For more information check [DVC documentation](https://dvc.org/doc/start/data-management?tab=Mac-Linux).

### Runing a model workflow

We currently use `immuneML` to run the models. In order to do so:

1. Generate the datasets of interest, store in `data`.
2. Generate `yaml` specification to describe the run. Check the existing files for examples. Use script_02_*.py. 
3. Run `immuneML` and store the results to be shared in `immuneML` directory. 

For more info, please check the [official immuneML docs](https://docs.immuneml.uio.no/latest/index.html#)

## Scripts

- script_01_build_datasets.py
- script_02_build_immuneML_spec.py
- script_03_run_immuneML.sh

## Notebooks

- 01_Explore.ipynb
- 02_Datasetgenerator.ipynb
- 03_Classifier_pairwise_datasets.ipynb
- 04_Pairwise_datasets.ipynb
- 05_1_vs_all_datasets.ipynb