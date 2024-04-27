"""
We verify dates of some attributions in the frozen dataset.
"""

import datetime
import json
import os
import sys
from pathlib import Path
from typing import List

from NegativeClassOptimization import datasets

# data_dir = Path("data/Frozen_MiniAbsolut_ML_shuffled")
data_dir = Path("data/Frozen_MiniAbsolut_ML")
REMOVE = False

## Verify mode
# "date": created/modifed; "attribution_type": "GLOBAL and LOCAL" attributions
# "attribution_exists": if {analysis_name} attribution json exists, and the weight
# "attribution_exists_simdis": same as before, for sim / dis analysis
VERIFY_MODE = "attribution_exists_simdis"  

if __name__ == "__main__":
    
    if VERIFY_MODE == "date":
        print(f"Verify attributions created/modified after 2023-12-01 in {data_dir}")

        # Traverse all the data_dir recursively, depth first search.
        filepaths = []
        for root, dirs, files in os.walk(data_dir, topdown=True):
            for name in files:
                path = os.path.join(root, name)
                # If files are found that have been created OR modified after 2023-12-01, collect the paths to these files.
                if os.path.getctime(path) > 1704812518 or os.path.getmtime(path) > 1704812518:
                    # Save filepath \t datetime created \t datetime modified
                    # Convert timestamp to datetime: datetime.datetime.fromtimestamp(1701442478)
                    filepaths.append(
                        f"{path}\t{datetime.datetime.fromtimestamp(os.path.getctime(path))}\t{datetime.datetime.fromtimestamp(os.path.getmtime(path))}"
                    )
                    if REMOVE:
                        os.remove(path)
        
        # Sort
        filepaths.sort()

        # Write the filepaths to a file.
        with open("data/paths_to_files_to_check.txt", "w") as f:
            for path in filepaths:
                f.write(path + "\n")
    
    elif VERIFY_MODE == "attribution_type":
        print(f"Verify attributions with GLOBAL and LOCAL attribution types in {data_dir}")

        # Traverse all the data_dir subdirectories recursively, depth first search.
        subdirectories = [x[0] for x in os.walk(data_dir)]
        for subdirectory in subdirectories:
                path = Path(subdirectory)

                if path.name == "v2.0-2":

                    # If no attribution_templates.json file is found, skip the directory.
                    if not (path / "attributor_templates.json").exists():
                        raise FileNotFoundError(f"No attributor_templates.json file found at {path}")
                    # Read attribution_templates.json
                    with open(path / "attributor_templates.json", "r") as f:
                        attribution_templates = f.read()
                        if len(attribution_templates) == 2:
                            print(f"Less 2 templates in attributor_templates.json at {path}")
                            continue
    
    elif VERIFY_MODE == "attribution_exists":
        
        from script_15_compute_attributions import (
            loader, task_generator_for_epitopes)

        # attr_analysis_name = "v2.0-3-epi"
        attr_analysis_name = "v2.0-2"

        tasks = task_generator_for_epitopes()  # note directory hardcoded in the other script!
        tasks = list(set([t for t in tasks]))

        for task in tasks:

            loader.load(task)
            bp = task.basepath
            fp = bp / f"attributions/{attr_analysis_name}/attribution_records.json"
            # Load json
            if not fp.exists():
                print(f"{task}: attribution_records.json not found at {fp}")
            else:
                with open(fp, "r") as f:
                    attr_json = json.load(f)
                    print(f"{task}: {len(attr_json)} records")
    
    elif VERIFY_MODE == "attribution_exists_simdis":

        from script_15_compute_attributions import (
            loader, task_generator_for_similar_or_dissimilar)

        attr_analysis_name = "v2.0-2"

        tasks = task_generator_for_similar_or_dissimilar(similar=True)  # note directory hardcoded in the other script!
        tasks = list(filter(lambda x: x.task_type in [
            datasets.ClassificationTaskType.HIGH_VS_LOOSER,
            datasets.ClassificationTaskType.HIGH_VS_95LOW,
        ], tasks))
        tasks = list(set([t for t in tasks]))

        for task in tasks:

            loader.load(task)
            bp = task.basepath
            fp = bp / f"attributions/{attr_analysis_name}/attribution_records.json"
            # Load json
            if not fp.exists():
                print(f"{task}: attribution_records.json not found at {fp}")
            else:
                with open(fp, "r") as f:
                    attr_json = json.load(f)
                    print(f"{task}: {len(attr_json)} records")
    