"""
Run motif discovery with XSTREME from MEME Suite.
Use python to run the XSTREME command line tool from docker container.
"""

import itertools
import logging
from pathlib import Path
from typing import Optional
import docker
from dataclasses import dataclass
from NegativeClassOptimization import config


TEST = False


@dataclass
class XSTREME_CMD:
    """
    Class that runs XSTREME from MEME Suite.
    """
    filepath_in: str  # Path to fasta file
    filepath_out: str  # Path to output directory
    background: Optional[str] = None  # Path to background fasta file


    def as_str(self):
        """
        Return XSTREME command as string.
        """
        cmd = f"xstreme --p {self.filepath_in} -oc {self.filepath_out} --protein --minw 4 --maxw 11"
        if self.background:
            cmd += f" --n {self.background}"
        return cmd


    def run(self):
        """
        Run XSTREME command.
        """
        client = docker.from_env()
        client.containers.run(
            "memesuite/memesuite",
            self.as_str(),
            volumes={
                str(config.DATA_BASE_PATH.parent): {
                    "bind": "/home/meme",
                    "mode": "rw",
                }
            },
            working_dir="/home/meme",
            remove=True,
            detach=False,
            user="meme",
            name="nco_xstreme",
        )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("data/logs/xstreme.log"),
        logging.StreamHandler(),
    ],
)


if __name__ == "__main__":

    if TEST:
        # Run XSTREME
        cmd = XSTREME_CMD(
            filepath_in="data/MiniAbsolut/1ADQ/fasta/high_test_5000.fasta",
            filepath_out="data/MiniAbsolut/1ADQ/xstreme/high_test_5000",
        )
        cmd.run()

    else:        
        ### Assemble commands
        cmds = []

        ## Shuffled background
        out_dir = config.DATA_XSTREME / "shuffled_background"
        if not out_dir.exists():
            out_dir.mkdir()
        for ag in config.ANTIGENS:
            ag_fasta_dir = Path(f"data/MiniAbsolut/{ag}/fasta")
            # Do it only for test datasets
            for fp in ag_fasta_dir.glob("*test*.fasta"):
                filepath_in = fp
                # Need relative path for docker container
                # Get path relative to data/
                filepath_out = out_dir / f"{ag}_{fp.stem}"
                filepath_out = filepath_out.relative_to(config.DATA_BASE_PATH.parent)
                cmd = XSTREME_CMD(
                    filepath_in=str(filepath_in),
                    filepath_out=str(filepath_out),
                )
                cmds.append(cmd)

        ## 1 vs 1
        out_dir = config.DATA_XSTREME / "1vs1"
        if not out_dir.exists():
            out_dir.mkdir()
        # build 2-permutations
        perms = itertools.permutations(config.ANTIGENS, r=2)
        for perm in perms:
            ag1, ag2 = perm
            ag1_fasta = Path(f"data/MiniAbsolut/{ag1}/fasta/high_test_5000.fasta")
            ag2_fasta = Path(f"data/MiniAbsolut/{ag2}/fasta/high_test_5000.fasta")
            filepath_out = out_dir / f"{ag1}_vs_{ag2}"
            filepath_out = filepath_out.relative_to(config.DATA_BASE_PATH.parent)
            cmd = XSTREME_CMD(
                filepath_in=str(ag1_fasta),
                filepath_out=str(filepath_out),
                background=str(ag2_fasta),
            )
            cmds.append(cmd)
        
        # high vs looser
        out_dir = config.DATA_XSTREME / "high_vs_looser"
        if not out_dir.exists():
            out_dir.mkdir()
        for ag in config.ANTIGENS:
            ag_fasta_dir = Path(f"data/MiniAbsolut/{ag}/fasta")
            # Do it only for test datasets
            filepath_in = ag_fasta_dir / "high_test_5000.fasta"
            background_fasta = ag_fasta_dir / "looserX_test_5000.fasta"
            filepath_out = out_dir / f"{ag}"
            filepath_out = filepath_out.relative_to(config.DATA_BASE_PATH.parent)
            cmd = XSTREME_CMD(
                filepath_in=str(filepath_in),
                filepath_out=str(filepath_out),
                background=str(background_fasta),
            )
            cmds.append(cmd)

        # high vs 95low
        out_dir = config.DATA_XSTREME / "high_vs_95low"
        if not out_dir.exists():
            out_dir.mkdir()
        for ag in config.ANTIGENS:
            ag_fasta_dir = Path(f"data/MiniAbsolut/{ag}/fasta")
            # Do it only for test datasets
            filepath_in = ag_fasta_dir / "high_test_5000.fasta"
            background_fasta = ag_fasta_dir / "95low_test_5000.fasta"
            filepath_out = out_dir / f"{ag}"
            filepath_out = filepath_out.relative_to(config.DATA_BASE_PATH.parent)
            cmd = XSTREME_CMD(
                filepath_in=str(filepath_in),
                filepath_out=str(filepath_out),
                background=str(background_fasta),
            )
            cmds.append(cmd)

        # Run sequentially
        logging.info(f"Assembled {len(cmds)} commands")
        # logging.info(f"Commands: {[cmd.as_str() for cmd in cmds]}")
        for cmd in cmds:
            logging.info(f"Running `{cmd.as_str()}`")
            cmd.run()
            logging.info(f"Finished `{cmd.as_str()}`")
