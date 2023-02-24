from copy import deepcopy

import NegativeClassOptimization.datasets as datasets
from NegativeClassOptimization.preprocessing import onehot_encode_df

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import math
from tqdm.notebook import tqdm
import json
from pathlib import Path

ONE_VS_ALL_PATH = Path('../data/full_data/1_vs_all')
ONE_VS_MIX_ALL_PATH = Path('../data/full_data/mix_all')
PAIRWISE_HIGH_PATH = Path('../data/full_data/high_pairwise')
HIGH_LOW_PATH = Path('../data/full_data/high_low_concat')


antigens = ['3VRL', '1NSN', '3RAJ', '5E94', '1H0D', '1WEJ', '1ADQ', '1FBI','2YPV', '1OB1']

#probably not needed 
class UnitNormLayer(nn.Module):
    def __init__(self):
        super(UnitNormLayer, self).__init__()
    def forward(self,x):
        norm = torch.norm(x, dim=0) #chamged dim=1
        return x / norm #torch.reshape(x, [-1,1])
    
def to_BinaryDataset(df):
    df["X"] = df['Slide_onehot']
    df["y"] = df['binder'].astype(int)
    df = datasets.BinaryDataset(df)
    return df

def dataset_preprosessing(df, scale = False):
    df = onehot_encode_df(df)

    #scale (why to scale one-hot?)
    if scale:
        arr_from_series = lambda s: np.stack(s, axis=0)

        onehot_stack = arr_from_series(df["Slide_onehot"])
        scaler = StandardScaler()
        df["Slide_onehot"] = scaler.fit_transform(onehot_stack).tolist()
        
    df = to_BinaryDataset(df)
    return df

def train_val_test_prep(df, scale = False):
    #df = onehot_encode_df(df)
    df_train = deepcopy(df[df['Train'] == True]).reset_index(drop=True)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)
    df_test = deepcopy(df[df['Train'] == False]).reset_index(drop=True)
    
    df_train = dataset_preprosessing(df_train, scale=scale)
    df_val = dataset_preprosessing(df_val, scale=scale)
    df_test = dataset_preprosessing(df_test, scale=scale)

    return df_train, df_val, df_test


def compute_binary_metrics(y_test_pred, y_test_true) -> dict:
    acc = metrics.accuracy_score(y_true=y_test_true, y_pred=y_test_pred)
    recall= metrics.recall_score(y_true=y_test_true, y_pred=y_test_pred)
    prec = metrics.precision_score(y_true=y_test_true, y_pred=y_test_pred)
    f1 = metrics.f1_score(y_true=y_test_true, y_pred=y_test_pred)
    return [acc, recall, prec, f1]

def test_loop(model, X, y, device,loss_fn, compute_loss = True,compute_merics = True, return_class = False):
    X = X.to(device)
    y = y.to(device)
    model = model.to(device)
    with torch.no_grad():
        y_pred, logits = model(X, return_logits = True)
        if return_class:
            bi_class = y_pred.numpy()[0][0]
            return round(bi_class)
        if compute_loss:
            loss = loss_fn(y_pred, y).cpu().detach().numpy()
            performance = [loss]
        if compute_merics:
            y_pred = y_pred.cpu().detach().numpy().reshape(-1).round()
            y = y.cpu().detach().numpy()
            performance = performance + compute_binary_metrics(y_pred, y)
    return performance

def get_open_closed_dataloader(open_path, closed_path):
    df_open = pd.read_csv(open_path, sep='\t') #HIGH_LOW_PATH / f'{ag}_high_low.tsv'
    df_open = df_open[df_open['Train'] == False].reset_index(drop =True)
    df_open = dataset_preprosessing(df_open)
    open_dataloader = DataLoader(df_open, batch_size=df_open.__len__(), shuffle=True)
    
    df_closed = pd.read_csv(closed_path, sep='\t')#ONE_VS_ALL_PATH / f'{ag}_vs_all.tsv'
    df_closed = df_closed[df_closed['Train'] == False].reset_index(drop =True)
    df_closed = dataset_preprosessing(df_closed)
    closed_dataloader = DataLoader(df_closed, batch_size=df_closed.__len__(), shuffle=True)
    return open_dataloader, closed_dataloader

def get_open_closed_perf(model, open_dataloader, closed_dataloader):
    loss_fn = nn.BCELoss()
    device = 'cpu'
    open_performance = dict()
    for batch, (X, y) in enumerate(open_dataloader): 
        open_performance = test_loop(model, X, y,device,loss_fn)
    open_performance[0] = float(np.mean(open_performance[0]))
        
    closed_performance = dict()
    for batch, (X, y) in enumerate(closed_dataloader): 
        closed_performance = test_loop(model, X, y, device,loss_fn )
    closed_performance[0] = float(np.mean(closed_performance[0])) 

    return open_performance, closed_performance

class SN10_stab(nn.Module):

    def __init__(self, normalize=False):
        super(SN10_stab, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(11*20, 10)
        nn.init.kaiming_normal_(self.linear1.weight)
        self.linear_relu_stack = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.normalize = normalize 
        if self.normalize:
            self.norm = UnitNormLayer()
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def forward(
        self, 
        x: torch.Tensor, 
        return_logits = False
        ):
        logits = self.forward_logits(x)
        if self.normalize:
            logits = self.norm(logits)
        expits = self.sigmoid(logits)
        if return_logits:
            return expits, logits
        else:
            return expits
        
"""def train_SN10(train_loader, val_loader, file_name, epochs=100, learning_rate = 0.002, momentum = 0, weight_decay = 0, patience = 10, min_delta = 0.005):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    baseline_model = SN10().to(device)

    optimizer = torch.optim.Adam(
                baseline_model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay,
                )
    loss_fn = nn.BCELoss()

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        #train
        running_loss = []
        for batch, (X, y) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            X = X.to(device)
            y = y.to(device)
            y_pred, logits = baseline_model(X, return_logits = True)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        train_loss = np.mean(running_loss)
        train_losses.append(float(train_loss))

        #validation
        running_loss = []
        with torch.no_grad():
            validation_loss = []
            for batch, (X,y) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                X = X.to(device)
                y = y.to(device)
                y_pred = baseline_model(X)
                running_loss.append(loss_fn(y_pred, y).cpu().detach().numpy())
            val_loss = np.mean(running_loss) 
            val_losses.append(float(val_loss))
        print(f"Epoch: {epoch+1}/{epochs} - Train loss: {train_loss}, Validation loss: {val_loss}") #save to log-file
        with open(f'./logs/{file_name}.txt', 'a') as log:
            log.write(f"Epoch: {epoch+1}/{epochs} - Train loss: {train_loss}, Validation loss: {val_loss}\n")

        #alternative early stopping
        if len(val_losses) >= patience:
            last_losses = val_losses[-patience:]
            max_loss = max(last_losses)
            min_loss = min(last_losses)
            if min_loss + min_delta >= max_loss:
                print("Early stopping") #save-to log file
                #save train_losses and val_losses to log file
                #save the model
                break
    with open(f'./logs/{file_name}_train_val_losses.json', 'w') as f:
        json.dump([train_losses, val_losses], f)
    torch.save(baseline_model.state_dict(), f'./torch_models/SN10_1_vs_all/{file_name}.pt')
    
    
class SN10(nn.Module):
    #The simple neural network 10 (SN10) model from `Absolut!`

    def __init__(self, normalize=False):
        super(SN10, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11*20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.normalize = normalize 
        if self.normalize:
            self.norm = UnitNormLayer()
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def forward(
        self, 
        x: torch.Tensor, 
        return_logits = False
        ):
        logits = self.forward_logits(x)
        if self.normalize:
            logits = self.norm(logits)
        expits = self.sigmoid(logits)
        if return_logits:
            return expits, logits
        else:
            return expits
            """

