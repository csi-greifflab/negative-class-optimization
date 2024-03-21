from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import numpy as np
import os

def blob_filename(nobs, nclasses, ndims):
    return f"blobs_obs{nobs}_classes{nclasses}_dims{ndims}"

if __name__ == "__main__":
    print("Blobs datasets will be created at data/blob_obs{}_classes{}_dims{}")

    data_dir = "../data/"

    os.makedirs(data_dir, exist_ok=True)

    num_obs = [5000]
    nclasses = [2, 10]
    dims = [100, 700]

    print("Current parameters:")
    print(f"Number of observations: {num_obs}")
    print(f"Number of classes: {nclasses}")
    print(f"Number of dimensions: {dims}")
    print(f"Total number of datasets (nclasses*ndimensions) {len(nclasses)*len(dims)}")

    for o in num_obs:
        for c in nclasses:
            for d in dims:
                print(f"Creating {o} observations with {c} classes and {d} dimensions.")
                X, y = make_blobs(n_samples = o, n_features=d,
                    centers=c, cluster_std=1, random_state=42)
                scaler = MinMaxScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                X = X.astype('float32')
                outX = os.path.join(data_dir, blob_filename(o, c, d))
                outy = os.path.join(data_dir, blob_filename(o, c, d))
                os.makedirs(outX, exist_ok=True)
                os.makedirs(outy, exist_ok=True)
                np.save(os.path.join(outX, "X.npy"), X)
                np.save(os.path.join(outy, "y.npy"), y)
          

