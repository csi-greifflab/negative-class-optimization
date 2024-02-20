"""
Short helper script to zip a list of paths into an archive.
"""

import os
import zipfile

# List of paths to zip: files and directories
paths = [
    "data/MiniAbsolut/HR2P/",
    "data/Frozen_MiniAbsolut_ML/high_vs_looser/seed_0/split_42/HR2P_high__vs__HR2P_looser",
    "data/Frozen_MiniAbsolut_ML/high_vs_95low/seed_0/split_42/HR2P_high__vs__HR2P_95low",
    "data/Frozen_MiniAbsolut_ML/1v1/seed_0/split_42/HR2P__vs__HR2PIR",
]

# Output archive path
output_path = "data/HR2P_experimental_attributions.zip"


if __name__ == "__main__":
    # Create a new archive
    with zipfile.ZipFile(output_path, "w") as archive:
        for path in paths:
            if os.path.isfile(path):
                archive.write(path, os.path.basename(path))
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        archive.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, "..")))

    # Print the output path with size in MB for convenience
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Archive size: {size_mb:.2f} MB created at {output_path}")
