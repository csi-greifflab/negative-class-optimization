"""
Generate yaml specification file for running immuneML analyses.
"""

from pathlib import Path
import yaml
import logging
from docopt import docopt

import NegativeClassOptimization.config as config
import NegativeClassOptimization.immuneml_utils as immune_utils

logging.basicConfig(level=logging.DEBUG)

docopt_doc = """Build immuneML specifications.

Usage:
    script_02_build_immuneML_spec.py pairwise
    script_02_build_immuneML_spec.py 1_vs_all    

Options:
    -h --help   Show help.

"""


class NoAliasDumper(yaml.SafeDumper):
    """
    Yaml dumper without yaml references.
    """
    def ignore_aliases(self, data):
        return True


def basic_spec_from_dataset_paths(datasets_paths) -> dict:
    datasets = {}
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
    specification = immuneml_utils.ImmuneMLSpecBuilder(datasets).spec
    return specification


if __name__ == "__main__":

    arguments = docopt(docopt_doc, version='Naval Fate 2.0')

    if arguments["pairwise"]:
        logging.info("Building spec for pairwise analysis.")
        spec_fp = "immuneML/pairwise_analysis_1.yml"
        datasets_paths = list(Path("./data/pairwise").glob("*.tsv"))
    elif arguments["1_vs_all"]:
        logging.info("Building spec for 1_vs_all analysis.")
        spec_fp = "immuneML/1_vs_all_analysis_1.yml"
        datasets_paths = list(Path("./data/1_vs_all").glob("*.tsv"))
    
    specification = basic_spec_from_dataset_paths(datasets_paths)
    with open(spec_fp, 'w+') as f:
        yaml.dump(
            specification,
            f,
            default_flow_style=False,
            sort_keys=False,
            Dumper=NoAliasDumper
        )
