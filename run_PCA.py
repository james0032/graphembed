import argparse
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sbn
sbn.set_style("white")

def run_pca(emb, n_pca):
    # Perform PCA
    pca = PCA(n_components=n_pca)  # Keep all components
    principle_components = pca.fit_transform(emb)
    return pca

def savefig(args):
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].bar(range(1,(args.ncomp+1)), gswithbert_pca.explained_variance_ratio_)
    axes[1].bar(range(1,(args.ncomp+1)), np.cumsum(gswithbert_pca.explained_variance_ratio_))
    axes[0].set_xlabel("Pricipal component")
    axes[0].set_ylabel("Explained Variance")
    axes[0].set_ylim(0,1)
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("Cumulative explained Variance")
    fig.suptitle(args.title)
    sbn.despine()
    fig.tight_layout()
    plt.savefig(args.outfig, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-emd_pkl", type=str)
    parser.add_argument("-ncomp", type=int, default=10)
    parser.add_argument("-outfig", type=str)
    parser.add_argument("-fig_title", type=str)
    args = parser.parse_args()
    with open(args.emd_pkl, "rb") as f:
        emb_vector = pickle.load(f)
    gswithbert_pca = run_pca(np.array(list(emb_vector.values())), args.ncomp)
    savefig(args)
    