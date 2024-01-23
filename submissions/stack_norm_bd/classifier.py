import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.simplefilter('ignore')


class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        estimators = [
            ('ridge', RidgeClassifier(random_state=42)),
            ('sgd', SGDClassifier(random_state=42)),
            ('mlp', make_pipeline(
                StandardScaler(),
                MLPClassifier(random_state=42)
            )),
            ('xrf', ExtraTreesClassifier(n_estimators=200, random_state=42)),
            ('lgbm', LGBMClassifier(
                learning_rate=0.5, verbose=-100, n_jobs=-1))
        ]

        self.pipe = make_pipeline(
            StackingClassifier(estimators=estimators,
                               final_estimator=make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', solver='saga')))
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
    # check if there is nan value
    if np.isnan(devi_data).sum() is not None:
        devi_data[np.isnan(devi_data)] = 0

    # Identify the highly deviance genes indices
    indices = np.argsort(devi_data)[::-1][:num_genes]

    return indices
# top 500
# CV fold 0
#         score  bal_acc        time
#         train     1.00  218.127125
#         valid     0.81    0.164999
#         test      0.82    0.102002
# CV fold 1
#         score  bal_acc        time
#         train     1.00  228.989447
#         valid     0.83    0.174999
#         test      0.81    0.108001
# CV fold 2
#         score  bal_acc        time
#         train     1.00  226.059481
#         valid     0.86    0.181001
#         test      0.80    0.108005
# CV fold 3
#         score  bal_acc        time
#         train     1.00  226.676614
#         valid     0.87    0.159050
#         test      0.82    0.109577
# CV fold 4
#         score  bal_acc        time
#         train     1.00  227.525819
#         valid     0.86    0.168102
#         test      0.86    0.110052
# ----------------------------
# Mean CV scores
# ----------------------------
#         score        bal_acc          time
#         train     1.0 +- 0.0  225.5 +- 3.8
#         valid  0.85 +- 0.023   0.2 +- 0.01
#         test   0.82 +- 0.018    0.1 +- 0.0
# ----------------------------
# Bagged scores
# ----------------------------
#         score  bal_acc
#         valid     0.85
#         test      0.84
###########################
# top 4000
# CV fold 0
#         score  bal_acc        time
#         train     1.00  817.018886
#         valid     0.85    0.275625
#         test      0.83    0.149999
# CV fold 1
#         score  bal_acc        time
#         train     1.00  843.302340
#         valid     0.86    0.280527
#         test      0.84    0.152997
# CV fold 2
#         score  bal_acc        time
#         train     1.00  858.673223
#         valid     0.87    0.286443
#         test      0.82    0.161527
# CV fold 3
#         score  bal_acc        time
#         train     1.00  839.426921
#         valid     0.84    0.291030
#         test      0.83    0.164003
# CV fold 4
#         score  bal_acc        time
#         train     1.00  863.623369
#         valid     0.87    0.295444
#         test      0.83    0.164110
# ----------------------------
# Mean CV scores
# ----------------------------
#         score        bal_acc            time
#         train     1.0 +- 0.0  844.4 +- 16.43
#         valid  0.86 +- 0.012     0.3 +- 0.01
#         test   0.83 +- 0.006     0.2 +- 0.01
# ----------------------------
# Bagged scores
# ----------------------------
#         score  bal_acc
#         valid     0.87
#         test      0.83
################################
# +SDGC 4000g
# CV fold 0
#         score  bal_acc       time
#         train     1.00  26.891914
#         valid     0.83   0.241524
#         test      0.83   0.137517
# CV fold 1
#         score  bal_acc       time
#         train     1.00  27.443138
#         valid     0.86   0.250254
#         test      0.83   0.138098
# CV fold 2
#         score  bal_acc       time
#         train     1.00  28.828294
#         valid     0.87   0.253042
#         test      0.82   0.145004
# CV fold 3
#         score  bal_acc       time
#         train     1.00  27.018362
#         valid     0.84   0.258563
#         test      0.83   0.139002
# CV fold 4
#         score  bal_acc       time
#         train     1.00  27.693832
#         valid     0.88   0.261816
#         test      0.85   0.143517
# ----------------------------
# Mean CV scores
# ----------------------------
#         score        bal_acc          time
#         train     1.0 +- 0.0  27.6 +- 0.69
#         valid  0.86 +- 0.018   0.3 +- 0.01
#         test   0.83 +- 0.012    0.1 +- 0.0
# ----------------------------
# Bagged scores
# ----------------------------
#         score  bal_acc
#         valid     0.86
#         test      0.85
#############################
# +Ridge
# CV fold 0
#         score  bal_acc       time
#         train     1.00  29.061688
#         valid     0.86   0.249771
#         test      0.85   0.141000
# CV fold 1
#         score  bal_acc       time
#         train     1.00  28.756564
#         valid     0.87   0.255048
#         test      0.84   0.138999
# CV fold 2
#         score  bal_acc       time
#         train     1.00  28.754598
#         valid     0.87   0.258827
#         test      0.82   0.141005
# CV fold 3
#         score  bal_acc       time
#         train     1.00  30.648854
#         valid     0.84   0.266045
#         test      0.85   0.143996
# CV fold 4
#         score  bal_acc       time
#         train     1.00  29.402581
#         valid     0.88   0.255993
#         test      0.86   0.153005
# ----------------------------
# Mean CV scores
# ----------------------------
#         score        bal_acc         time
#         train     1.0 +- 0.0  29.3 +- 0.7
#         valid  0.86 +- 0.013  0.3 +- 0.01
#         test   0.84 +- 0.012   0.1 +- 0.0
# ----------------------------
# Bagged scores
# ----------------------------
#         score  bal_acc
#         valid     0.87
#         test      0.86
