import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster import SpectralClustering

def ReadData(path, normalize=False) :
    # Reading the file
    df = pd.read_csv(path)
    # Dropping the month column
    df = df.drop(columns=["mnth", "total_liability", "total_cashsetoff", "total_itc_claimed"], axis=1)
    # Grouping by id and taking the mean
    df = df.groupby(["id"]).mean().drop_duplicates()
    # Normalizing the data
    if normalize is True:
        df = (df - df.mean()) / df.std()
    return df.to_numpy()

def GetEigenValues(matrix) :
    # Covariance matrix
    covar = matrix.T @ matrix
    # Eigenvalues
    values = eigh(covar, eigvals_only = True)
    return values

if __name__ == "__main__" :
    # Read data
    data = ReadData("./resources/data_class.csv")
    # Spectral Clustering
    clusters = OPTICS(min_samples=5, n_jobs=-1).fit(data)
    print(clusters.labels_)
