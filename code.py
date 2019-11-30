import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import eigh
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster import SpectralClustering, KMeans

def ReadData(path, normalize=False, drop=False, minmax=False) :
    # Reading the file
    df = pd.read_csv(path)
    # Dropping the month column
    df = df.drop(columns=["mnth", "total_liability", "total_cashsetoff", "total_itc_claimed"], axis=1)
    # Grouping by id and taking the mean
    df = df.groupby(["id"]).mean()
    # Dropping duplicate rows after reducing dimensions
    if drop is True :
        df = df.drop_duplicates()
    # Normalizing the data
    if normalize is True:
        df = (df - df.mean()) / df.std()
    # MinMax normalizing
    elif minmax is True :
        df = (df - df.min()) / (df.max() - df.min())

    return df.to_numpy()

if __name__ == "__main__" :
    # Read data
    data = ReadData("./resources/data_class.csv", minmax=True)

    # Fraud nodes
    isFraud = np.zeros(len(data))   

    # Statistics
    stats = []

    # Fraud threshold
    threshold = 100

    # Looping over different values
    for clusters in range(2, 16) :
        sc = SpectralClustering(n_clusters=clusters, n_jobs=-1).fit(data)
        counts = collections.Counter(sc.labels_)
        
        genuine = []
        for cluster, count in counts.most_common(clusters) :
            if count > threshold :
                genuine.append(cluster)

        # Classfying taxpayers
        fraud = 0
        for label, idx in zip(sc.labels_, range(len(data))) :
            if label not in genuine :
                isFraud[idx] = 1
                fraud += 1

        stats.append([clusters, fraud])

    stats = np.array(stats)
    print(pd.DataFrame(stats, columns=["Clusters", "Frauds found"]))

    # Plot
    y_pos = np.arange(len(stats))
    plt.bar(y_pos, stats[:, 1])
    plt.xticks(y_pos, stats[:, 0])
    plt.xlabel('Frauds Found')
    plt.title('Clusters vs frauds')
    plt.savefig("./images/plot.png")
    plt.show()

    # Write frauds found to a file
    frauds = 0
    with open("results.txt", "w") as f :
        for idx in range(len(isFraud)) :
            if isFraud[idx] == 1 :
                f.write("%d " %idx)
                frauds += 1
    print("%d frauds found." %frauds)

