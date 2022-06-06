"""
Generate yaml specification file for running immuneML analyses.
"""

from pathlib import Path
import yaml

import config
import immuneml_utils


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


if __name__ == "__main__":

    spec_fp = "immuneML/pairwise_analysis_1.yml"

    datasets = {}
    datasets_paths = list(Path("./data/pairwise").glob("*.tsv"))
    for dataset_path in datasets_paths:
        dataset_name = dataset_path.name.split(".")[0]
        dataset_path = str(dataset_path.absolute())
        datasets[dataset_name] = {
            "format": "Generic",
            "params": {
                "path": dataset_path,
                "is_repertoire": False,
                "region_type": "FULL_SEQUENCE",
                "column_mapping": {
                    "CDR3": "sequence_aas",
                },
                "metadata_column_mapping": {
                    "UID": "UID",
                    "Antigen": "Antigen",
                    "binder": "binder",
                }
            },
        }
    specification = immuneml_utils.ImmuneMLSpecBuilder(datasets).specs

    with open(spec_fp, 'w+') as f:
        yaml.dump(
            specification,
            f,
            default_flow_style=False,
            sort_keys=False,
            Dumper=NoAliasDumper
        )
