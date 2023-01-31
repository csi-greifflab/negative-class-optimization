import logging
from pathlib import Path
import pickle
from typing import List, Optional
from joblib import Parallel, delayed
from NegativeClassOptimization import utils
from NegativeClassOptimization import preprocessing


def save_dict_as_pickle(dict_, path):
    with open(path, "wb+") as f:
        pickle.dump(dict_, f)


def multiprocessing_wrapper_script_11(
    slides: List[str], 
    embedder,
    save_dir: Path,
    batch_number: Optional[int] = None,
    ):
    
    slide_embeddings_per_residue = {}
    slide_embeddings_per_prot = {}

    logging.info(f"Computing embeddings for batch {batch_number} with {type(embedder)}.")

    for slide in slides:
        
        emb = embedder.embed(slide)
        emb_per_prot = embedder.reduce_per_protein(emb)

        slide_embeddings_per_residue[slide] = emb.tolist()
        slide_embeddings_per_prot[slide] = emb_per_prot.tolist()
    
    save_dict_as_pickle(slide_embeddings_per_residue, save_dir / f"slide_embeddings_per_residue_b{batch_number}.pkl")
    save_dict_as_pickle(slide_embeddings_per_prot, save_dir / f"slide_embeddings_per_prot_b{batch_number}.pkl")


if __name__ == "__main__":
    
    logging.basicConfig(
        format="%(asctime)s %(process)d %(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename="data/logs/11.log",
                mode="a",
            ),
            logging.StreamHandler()
        ]
    )

    df = utils.load_global_dataframe()
    slide_lists = []
    slides = list(set(df["Slide"]))
    NUM_SEQ_PER_BATCH = 5000
    for i in range(0, len(slides), NUM_SEQ_PER_BATCH):
        slide_lists.append(slides[i : i + NUM_SEQ_PER_BATCH])

    logging.info("Computing embeddings for ProtTransT5XLU50.")
    pt_embedder = preprocessing.load_embedder("ProtTransT5XLU50")
    Parallel(n_jobs=10)(
        delayed(multiprocessing_wrapper_script_11)(
            slides, 
            pt_embedder, 
            Path("data/slack_1/global/embeddings/ProtTransT5XLU50"),
            i+1,
            ) for i, slides in enumerate(slide_lists)
    )

    logging.info("Computing embeddings for ESMB1b.")
    esm1b_embedder = preprocessing.load_embedder("ESM1b")
    Parallel(n_jobs=10)(
        delayed(multiprocessing_wrapper_script_11)(
            slides, 
            esm1b_embedder, 
            Path("data/slack_1/global/embeddings/ESM1b"),
            i+1,
            ) for i, slides in enumerate(slide_lists)
    )
