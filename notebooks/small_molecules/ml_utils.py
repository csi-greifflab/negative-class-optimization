from pathlib import Path

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

import pickle

import catboost

from rdkit import Chem
from rdkit.Chem import AllChem

def create_splits(df, path_to_save_splits, split_seeds, test_size):
    for split in split_seeds:
        split_path = Path(path_to_save_splits) / f'split_{split}'
        split_path.mkdir(parents=True, exist_ok=True)
        
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=split, stratify=df['Y_binary'])
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        # Save as pickle
        df_train.to_pickle(split_path / 'train.pkl')
        df_test.to_pickle(split_path / 'test.pkl')


def one_hot_encode(line, max_len, char_dict):
    """
    One hot encode a string of characters. 
    If the string is shorter than max_len it will be padded with zeros.

    Args:
    line: str, input string
    max_len: int, length of the output vector
    char_dict: dict, dictionary that maps each character to an integer
    """
    
    num_chars = len(char_dict)
    X = np.zeros((num_chars, max_len), dtype=np.float32)

    if type(line)!=str:
        print('format is not str!')
    for i, ch in enumerate(line[:max_len]):
        tmp = char_dict.get(ch)
        if tmp:
            X[tmp - 1, i] = 1
        else:
            print(line,'Unexpected character',ch)
    X = np.array(X) #.flatten() 
    return X


def smiles_to_ecfp8(smiles: str, n_bits: int = 4096) -> list[int]:
    """
    Convert SMILES string to ECFP8 fingerprint
    
    Parameters:
        smiles (str): Input SMILES string
        n_bits (int): Fingerprint length (default: 4096)
    
    Returns:
        list[int]: ECFP8 fingerprint as bit vector
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    return np.array(AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius=4,  # ECFP8 = 4-radius circular substructures
        nBits=n_bits,
        useChirality=True
    ), dtype=int)

def load_custom_data(pkl_path, X_col='X', y_col='Y_binary'):

    df = pd.read_pickle(pkl_path)
    X = np.array(df[X_col].tolist())
    y = np.array(df[y_col].tolist())
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

chemberta.eval()

def chembert_encode(df, SMILE_col, padding=True):

    smiles_list = df[SMILE_col].tolist()
    embeddings_cls = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)

    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
            model_output = chemberta(**encoded_input)
            
            embedding = model_output[0][::,0,::]
            embeddings_cls[i] = embedding
            
            embedding = torch.mean(model_output[0],1)
            embeddings_mean[i] = embedding
            
    return embeddings_cls.numpy(), embeddings_mean.numpy()

def train_catboost_by_task(X,y, path_to_save_model=None):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = catboost.CatBoostClassifier(
        iterations=500,
        learning_rate=0.01,
        loss_function='CrossEntropy',
        eval_metric='Accuracy',
        random_seed=42,
        l2_leaf_reg=3,
        bagging_temperature=1,
        random_strength=1,
        verbose=False
        )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    # save best model
    if path_to_save_model:
        model.save_model(path_to_save_model)

    return model

def load_trained_model(path_model):
    model = catboost.CatBoostClassifier()
    model.load_model(path_model)
    return model