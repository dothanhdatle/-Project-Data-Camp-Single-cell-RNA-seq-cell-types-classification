import multiprocessing as mp
import scipy.sparse
import scipy.stats
import numpy as np


def binomial_chunk(inds, Y, n):
    """
    Compute binomial deviance for a chunk of the count matrix.

    Returns a vector with the per-gene deviance of the chunk.

    Input:
    ------
            inds - a pair of integer indices, defining the start and end gene indices for the chunk
            Y - count matrix, raw counts, cells as rows, genes as columns
            n - n in the R implementation, for deviance computation
    """
    # subset input data to specified genes for the chunk
    Yn = Y[:, inds[0]: inds[1]]
    # set up appropriate dimensionality for p
    p = np.sum(Yn, axis=0) / np.sum(n)
    p = p[None, :]
    # in R, they skip all the nan values from the thing defined here as holder
    # seeing how it gets multiplied by the values from Y, instead of trying to find some weird skip
    # just set all the zero values at this step to 1. this way they become 0 after logging
    # and successfully go away in the multiplication. gives same results as matching R command
    holder = Yn / (n * p)
    holder[holder == 0] = 1
    term1 = np.sum(Yn * np.log(holder), axis=0)
    nx = n - Yn
    # same thing, second holder
    holder = nx / (n * (1 - p))
    holder[holder == 0] = 1
    term2 = np.sum(nx * np.log(holder), axis=0)
    return 2 * (term1 + term2)


# initiating global variables for the chunk processes to use
def _pool_init(_Y, _n, _chunk_func):
    global Y, n, chunk_func
    Y = _Y
    n = _n
    chunk_func = _chunk_func


# a function that calls the requisite chunk computation based on inds and global variables
def pool_func(inds):
    return chunk_func(inds, Y, n)


def binomial_deviance(Y, flavor='binomial', chunksize=1e8, n_jobs=None):
    """
    A function to compute the binomial deviance of a count matrix.

    Returns a vector of one deviance per gene

    Input:
    ------
            Y : ``numpy.array``
                    Observations as rows, features as columns, raw count (integer) data
            all other input as in highly_deviant_genes()
    """
    # if no info on n_jobs, just go for all the cores
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    # computation prep - obtain n(bin), store in variable called n regardless
    # and point chunk_func to appropriate function that computes the deviance of chunks
    if flavor == 'binomial':
        chunk_func = binomial_chunk
        n = np.sum(Y, axis=1)
    else:
        raise ValueError('incorrect flavor value')
    # ensure that n has the correct dimensionality, needs to be treated differently if sparse counts
    if scipy.sparse.issparse(Y):
        n = np.asarray(n)
    else:
        n = n[:, None]
    # how many genes per job? we're aiming for chunksize total values in memory at a time
    # so divide that by n_jobs and the number of cells to get genes per job
    chunkcount = np.ceil(chunksize / (Y.shape[0] * n_jobs))
    # set up chunkcount-wide index intervals for count matrix subsetting within the jobs
    inds = []
    for ind in np.arange(0, Y.shape[1], chunkcount):
        inds.append([np.int_(ind), np.int_(ind + chunkcount)])
    # set up the parallel computation pool with the count matrix and n
    # chunk_func is set to the corresponding Poisson/binomial computation function
    p = mp.Pool(n_jobs, _pool_init, (Y, n, chunk_func))
    deviance = p.map(pool_func, inds)
    # collapse output and return
    return np.hstack(deviance)
