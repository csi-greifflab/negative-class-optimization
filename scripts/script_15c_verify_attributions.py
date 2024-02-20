"""
We verify dates of some attributions in the frozen dataset.
"""

import datetime
import os
import sys
from pathlib import Path

data_dir = Path("data/Frozen_MiniAbsolut_ML_shuffled")
REMOVE = False
VERIFY_MODE = "attribution_type"  # "date": created/modifed; "attribution_type": "GLOBAL and LOCAL" attributions

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
                    if not (path / "attribution_templates.json").exists():
                        raise FileNotFoundError(f"No attribution_templates.json file found at {path}")
                    # Read attribution_templates.json
                    with open(path / "attribution_templates.json", "r") as f:
                        attribution_templates = f.read()
                        if len(attribution_templates) == 2:
                            print(f"Less 2 templates in attribution_templates.json at {path}")
                            continue