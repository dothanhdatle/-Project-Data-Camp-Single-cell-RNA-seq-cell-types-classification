import numpy as np
from scipy.stats import median_abs_deviation
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline


class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        self.pipe = make_pipeline(
            LGBMClassifier(learning_rate=0.5, verbose=-100, n_jobs=-1)
        )

    def fit(self, X_sparse, y):
        X, y = _preprocess_X(X_sparse, y, quality_control=True)
        self.pipe.fit(X, y)

        pass

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse, quality_control=False)
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
    X_norm = norm_log(X)
    # Add the total genes of each cell as new feature
    X_norm = np.column_stack((X_norm, X.sum(axis=1)))
    if y is not None:
        return X_norm, y
    else:
        return X_norm
