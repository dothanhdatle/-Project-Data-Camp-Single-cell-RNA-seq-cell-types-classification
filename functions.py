import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import median_abs_deviation


# sklearn module
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



# Quality Control
def outlier_cells(X, num_mads=5):
    outliers = ((X < np.median(X) - num_mads * median_abs_deviation(X)
                 )
                | (np.median(X) + num_mads * median_abs_deviation(X) < X))
    return outliers


# Normalization with shifted logarithm
def norm_log(X):
    # Calculate the total counts per cell
    total_counts_cell = np.sum(X, axis=1)

    # Calculate the median of total counts
    med_total_counts = np.median(total_counts_cell)

    # Scale the total counts
    scaled_counts = (X / total_counts_cell[:, np.newaxis]) * med_total_counts

    # Logarithm transformation
    return np.log1p(scaled_counts)

def binomial_deviance(X):
    # ensure that n has the correct dimensionality, needs to be treated differently if sparse counts
    n = np.sum(X, axis=1)
    n = n[:, None]
    # set up appropriate dimensionality for p
    p = np.sum(X, axis=0) / np.sum(n)
    p = p[None, :]

    holder = X / (n * p)
    holder[holder == 0] = 1
    term1 = np.sum(X * np.log(holder), axis=0)
    nx = n - X
    # same thing, second holder
    holder = nx / (n * (1 - p))
    holder[holder == 0] = 1
    term2 = np.sum(nx * np.log(holder), axis=0)
    return 2 * (term1 + term2)

# Feature selection
def highly_deviance_indices(data, top_proportion, num_genes):
    # compute binomial deviances on the original count matrix
    devi_data = binomial_deviance(data)

    if np.isnan(devi_data).sum() is not None:
        devi_data[np.isnan(devi_data)] = 0

    # Get the number of genes to select
    if top_proportion is not None:
        num_genes = int(top_proportion * len(devi_data))

    # Identify the highly deviance genes indices
    indices = np.argsort(devi_data)[::-1][:num_genes]

    return indices


def HVGs(data, top_proportion, num_genes, flavor='cell_ranger'):
    """
    Parameters:
        - top_proportion: percentages of top highly variable genes to keep (None or percentage)
        - num_genes: number of genes to keep (None or int)
        - flavor: 'seurat' or 'cell_ranger'. Determines the method used for calculating dispersion.
    """
    # Calculate mean and dispersion
    mean_expr = np.mean(data, axis=0)
    dispersion = np.var(data, axis=0) / \
        mean_expr if flavor == 'seurat' else np.var(data, axis=0)

    # Get the number of genes to select
    if top_proportion is not None:
        num_genes = int(top_proportion * data.shape[1])

    # Get indices of top highly variable genes
    hvg_indices = np.argsort(dispersion)[::-1][:num_genes]

    return hvg_indices


# Dimension Reduction
def UMAP_reduction(data, components=50):
    from umap import UMAP

    umap_model = UMAP(n_components=components, random_state=0)
    dimension_data = umap_model.fit_transform(data)

    return dimension_data

def TSNE_reduction(data, components=2):
    from sklearn.manifold import TSNE
    
    # init t-SNE 2d
    tsne = TSNE(n_components=components, random_state=0)
    data_sne = tsne.fit_transform(data)

    return data_sne


# Preprocessing

def _preprocess_X(X, y=None, quality_control=True):
    # cast a dense array
    if (quality_control == True) & (y is not None):
        # Quality control
        total_cell_counts = X.sum(axis=1)  # total counts per cell
        # Number of expressed genes in a cell
        number_genes_by_counts = (X > 0).sum(axis=1)
        log1p_total_cells_counts = np.log1p(total_cell_counts)
        log1p_number_genes_by_counts = np.log1p(number_genes_by_counts)

        # Identify the top 20 genes
        top20_genes = np.argsort(np.array(X.sum(axis=0)).ravel())[::-1][:20]
        # Calculate the total counts in the top 20 genes for each cell
        total_counts_top20_genes = np.sum(X[:, top20_genes], axis=1)
        # Calculate the percentage of counts in the top 20 genes for each cell
        pct_counts_top20_genes = total_counts_top20_genes / total_cell_counts
        # Identify and remove poor quality cells
        cells_remove = (outlier_cells(log1p_total_cells_counts, 5)
                        | outlier_cells(log1p_number_genes_by_counts, 5)
                        | outlier_cells(pct_counts_top20_genes, 5))
        X = X[(~cells_remove), :].copy()
        y = y[~cells_remove].copy()

    # Normalization the total counts per cell
    X = norm_log(X)

    if y is not None:
        return X, y
    else:
        return X
