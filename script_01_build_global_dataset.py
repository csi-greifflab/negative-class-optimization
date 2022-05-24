"""
Build a global dataset from a base path.
"""

from pathlib import Path
import config
import utils


dataset_path: Path = config.DATA_SLACK_1


if __name__ == "__main__":
    global_datasets_dir: Path = config.GLOBAL_DATASETS_DIR
    dataset_name = dataset_path.name
    df_global = utils.build_global_dataset(dataset_path)
    df_global.to_csv(global_datasets_dir / f"{dataset_name}_global.tsv", sep='\t')