import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.simplefilter('ignore')


class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        estimators = [('ridge', RidgeClassifier(random_state=42)),
                      ('sgd', SGDClassifier(random_state=42)),
                      ('lgr', LogisticRegression(solver='saga')),
                      ('xrf', ExtraTreesClassifier(random_state=42)),
                      ('gb', GradientBoostingClassifier(n_estimators=200,
                                                        learning_rate=0.01, max_features="sqrt")),
                      ('mlp', make_pipeline(StandardScaler(),
                       MLPClassifier(random_state=42))),
                      ('lgbm', LGBMClassifier(learning_rate=0.5,
                                              verbose=-100, n_jobs=-1, random_state=42))
                      ]

        self.pipe = make_pipeline(
            StackingClassifier(estimators=estimators,
                               final_estimator=GradientBoostingClassifier())
        )

    def fit(self, X_sparse, y):
        X, y = _preprocess_X(X_sparse, y, quality_control=True)
        # select the top 4000 informative genes
        self.idx = highly_deviance_indices(X, num_genes=4000)
        X = X[:, self.idx]
        self.pipe.fit(X, y)

        pass

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse, quality_control=False)
        X = X[:, self.idx]
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


def highly_deviance_indices(data, num_genes):
    # compute binomial deviances on the original count matrix
    devi_data = binomial_deviance(data)
    if np.isnan(devi_data).sum() is not None:
        devi_data[np.isnan(devi_data)] = 0

    # Identify the highly deviance genes indices
    indices = np.argsort(devi_data)[::-1][:num_genes]

    return indices

# CV fold 0
#         score  bal_acc       time
#         train     0.99  74.515320
#         valid     0.87   0.366000
#         test      0.85   0.181968
# CV fold 1
#         score  bal_acc       time
#         train     0.98  76.691211
#         valid     0.88   0.442979
#         test      0.85   0.190014
# CV fold 2
#         score  bal_acc       time
#         train     1.00  75.217571
#         valid     0.87   0.353000
#         test      0.82   0.183994
# CV fold 3
#         score  bal_acc       time
#         train     1.00  96.566553
#         valid     0.90   0.372220
#         test      0.85   0.187865
# CV fold 4
#         score  bal_acc       time
#         train     1.00  96.276811
#         valid     0.87   0.332998
#         test      0.85   0.186000
# ----------------------------
# Mean CV scores
# ----------------------------
#         score        bal_acc           time
#         train  0.99 +- 0.007  83.9 +- 10.29
#         valid  0.88 +- 0.011    0.4 +- 0.04
#         test   0.85 +- 0.011     0.2 +- 0.0
# ----------------------------
# Bagged scores
# ----------------------------
#         score  bal_acc
#         valid     0.89
#         test      0.88
