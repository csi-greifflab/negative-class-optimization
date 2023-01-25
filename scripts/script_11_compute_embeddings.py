from pathlib import Path
import tqdm
import pickle
from NegativeClassOptimization import utils
from NegativeClassOptimization import preprocessing


EMB_SAVE_DIR = Path("data/slack_1/global/embeddings")


def save_dict_as_pickle(dict_, path):
    with open(path, "wb+") as f:
        pickle.dump(dict_, f)


if __name__ == "__main__":
    df = utils.load_global_dataframe()

    esm1b_embedder = preprocessing.load_embedder("ESM1b")
    pt_embedder = preprocessing.load_embedder("ProtTransT5XLU50")

    slide_embeddings_per_residue = {}
    slide_embeddings_per_prot = {}

    for i, slide in enumerate(tqdm.tqdm(df["Slide"])):
        
        esm1b_emb = esm1b_embedder.embed(slide)
        esm1b_emb_per_prot = esm1b_embedder.reduce_per_protein(esm1b_emb)

        pt_emb = pt_embedder.embed(slide)
        pt_emb_per_prot = pt_embedder.reduce_per_protein(pt_emb)

        slide_embeddings_per_residue[slide] = {
            "ESM1b": esm1b_emb.tolist(),
            "ProtTransT5XLU50": pt_emb.tolist(),
        }
        slide_embeddings_per_prot[slide] = {
            "ESM1b": esm1b_emb_per_prot.tolist(),
            "ProtTransT5XLU50": pt_emb_per_prot.tolist(),
        }

        if i % 100 == 0:
            save_dict_as_pickle(slide_embeddings_per_residue, EMB_SAVE_DIR / "slide_embeddings_per_residue.pkl")
            save_dict_as_pickle(slide_embeddings_per_prot, EMB_SAVE_DIR / "slide_embeddings_per_prot.pkl")
    
    save_dict_as_pickle(slide_embeddings_per_residue, EMB_SAVE_DIR / "slide_embeddings_per_residue.pkl")
    save_dict_as_pickle(slide_embeddings_per_prot, EMB_SAVE_DIR / "slide_embeddings_per_prot.pkl")