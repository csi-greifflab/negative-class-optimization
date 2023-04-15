"""
Script that extracts from a tsv a column with sequences and writes to a fasta file.
Operates on data/MiniAbsolut in current hardcoded version.
"""

from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from NegativeClassOptimization import config, utils


def make_fasta(fp_in, fp_out):
    """
    Function that extracts from a tsv a column with sequences and writes to a fasta file.
    """
    df = pd.read_csv(fp_in, sep="\t")
    df = df[["Slide"]]

    records = []

    for i, row in df.iterrows():
        record = SeqRecord(Seq(row["Slide"]), id="Slide" + str(i), description="")
        records.append(record)

    SeqIO.write(records, fp_out, "fasta")


if __name__ == "__main__":

    antigens = config.ANTIGENS

    for ag in antigens:

        print(f"Processing {ag}...")
        ag_dir = Path(f"data/MiniAbsolut/{ag}")
        for fp_in in ag_dir.glob("*tsv"):

            print(f"Processing {fp_in}...")
            fasta_dir = Path(f"data/MiniAbsolut/{ag}/fasta")
            if not fasta_dir.exists():
                fasta_dir.mkdir()

            fp_out = fasta_dir / f"{fp_in.stem}.fasta"
            make_fasta(fp_in, fp_out)
