# Negative dataset optimization

![version badge](https://img.shields.io/badge/version-0.99-yellow)

This is the `Readme` for the `NegativeDatasetOptimization` project, internally abbreviated as `nco`. This is the closest representative term, as the scope of the project is dynamic. Follow this readme to setup everything and run.

## Setup

Once you've cloned the repository, run:

```
bash manage.sh install_env
conda activate nco
bash manage.sh update_env  # updates and installs local packages
dvc pull  # fetches all the data
```

### Library dependencies

To add a new library dependence, please add it manually in the `environment.yml` file and run `bash manage.sh update_env`. Don't forget to commit and push.

### Data

#### DVC

This will setup the environment and required data, including the `data` directory, with all required data.

The `dvc pull` will fetch all the data, and at times this can be a lot. Often one will want to work with just a single file, one can achieve that with `dvc get` (usually not `dvc import`, to implement read-only mode, since we don't want changes to standard files). For convenience `./manage.sh get_700k` fetches the 700k dataset. We are also using `dvc` data pipelines. Please familiarize yourself with the API and check the file for the details.

Once you've made changes to a `DVC`-tracked directory, please add the changes to `dvc` and `git` (provided example is for the `data` directory):

```
dvc add data
git commit data.dvc -m "data directory updates"
dvc push
```

Note that to control file size, some large files are ignored (check `.dvcignore`).

For more information check [DVC documentation](https://dvc.org/doc/start/data-management?tab=Mac-Linux).

#### Download `Absolut` data
Get the doi csv from [data source](https://archive.norstore.no/pages/public/datasetDetail.jsf?id=10.11582/2021.00063), save it to data/Absolut/toc_doi10.11582_2021.00063.csv. Then run `python scripts/script_01_build_datasets.py download_absolut`.

### mlflow

We use `mlflow` to track experimental results, it is setup as a docker container. All details in `mlflow/` directory.

## Organization

We organize most of the code in:

1. Local library `src/NegativeClassOrganization` for functionality required everywhere else. Most organized and clean from this repo.
2. `/notebooks` for exploratory and results analyses, local development, short experiments and others in this spirit. The messiest fom this repo. We try to keep notebooks short and move well-established functionality into local library and scripts.
3. `/scripts` for the stable, established, reproducible analyses. These are tracked and managed through mlops tools like dvc and mlflow. They are cleaner than notebooks and leverage the local library. Check `dvc dag` for a dependency graph of the stages. **TODO:** add embed Google Sheets.
4. Other, such as `mlflow/` for docker-based deployment of `mlflow` and auxiliary services.
