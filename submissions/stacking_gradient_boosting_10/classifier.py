import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#import warnings
#warnings.simplefilter('ignore')


class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        estimators = [
          ('ridge', RidgeClassifier(random_state=42)),
          ('sgd', SGDClassifier(random_state=42)),
          ('mlp', make_pipeline(StandardScaler(), MLPClassifier(random_state=42))),
          ('xrf', ExtraTreesClassifier(n_estimators=100, random_state=42)), 
          ('gb', GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_features="log2")),
          ('lgbm', LGBMClassifier(learning_rate=0.5, verbose=-100, n_jobs=-1))
        ]

        self.pipe = make_pipeline(
            StackingClassifier(estimators=estimators,
                               final_estimator=make_pipeline(StandardScaler(), LogisticRegression()))
        )

    def fit(self, X_sparse, y):
        X, y = _preprocess_X(X_sparse, y, quality_control=True)
        self.ind = HVGs(X, top_proportion = 0.1, num_genes=None)
        X = X[:, self.ind]
        self.pipe.fit(X, y)

        pass

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse, quality_control=False)
        X = X[:, self.ind]
        return self.pipe.predict_proba(X)


# function to identify poor quality cells


def outlier_cells(X, num_mads=5):
    outliers = ((X < np.median(X) - num_mads * median_abs_deviation(X)
                 )
                | (np.median(X) + num_mads * median_abs_deviation(X) < X))
    return outliers


# function to normalize data


def norm_log(X):
    # Calculate the total counts per cell
    total_counts_cell = np.sum(X, axis=1)

    # Calculate the median of total counts
    med_total_counts = np.median(total_counts_cell)

    # Scale the total counts
    scaled_counts = (X / total_counts_cell[:, np.newaxis]) * med_total_counts

    # Logarithm transformation
    return np.log1p(scaled_counts)


def _preprocess_X(X_sparse, y=None, quality_control=True):
    # cast a dense array
    X = X_sparse.toarray()
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

# Feature selection

def HVGs(data, top_proportion, num_genes, flavor='cell_ranger'):
    """
    Parameters:
        - top_proportion: percentages of top highly variable genes to keep (None or percentage)
        - num_genes: number of genes to keep (None or int)
        - flavor: 'seurat' or 'cell_ranger'. Determines the method used for calculating dispersion.
    """
    # Calculate mean and dispersion
    mean_expr = np.mean(data, axis=0)
    dispersion = np.var(data, axis=0) / mean_expr if flavor == 'seurat' else np.var(data, axis=0)

    # Get the number of genes to select
    if top_proportion is not None:
        num_genes = int(top_proportion * data.shape[1])

    # Get indices of top highly variable genes
    hvg_indices = np.argsort(dispersion)[::-1][:num_genes]

    return hvg_indices