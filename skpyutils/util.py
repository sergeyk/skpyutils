"""
Helpful classes and methods for everyday Python usage.
Sergey Karayev - http://sergeykarayev.com
"""

import os
import sys
import subprocess
import operator
import time
import json
import numpy as np
import scipy.stats as st
from skpyutils.table import Table


class Report:
    """
    Convenience class to simplify aggregating reports to write to file.
    """
    def __init__(self):
        self.report = ''

    def append(self, str):
        "Print str and append it to the report so far as a new line."
        print(str)
        self.report += str + '\n'

    def __repr__(self):
        return self.report


def load_json(filename):
    assert(os.path.exists(filename))
    with open(filename) as f:
        data = json.load(f)
    return data


def append_index_column(arr, index):
    """
    Take an m x n array, and appends a column containing index.
    """
    ind_vector = np.ones((np.shape(arr)[0], 1)) * index
    arr = np.hstack((arr, ind_vector))
    return arr


def filter_on_column(arr, ind, val, op=operator.eq, omit=False):
    """
    Returns the rows of arr where arr[:,ind]==val,
    optionally omitting the ind column.
    """
    try:
        arr = arr[op(arr[:, ind], val), :]
    except:
        return arr
    # TODO: fix this mess
    if omit:
        final_ind = range(np.shape(arr)[1])
        final_ind = np.delete(final_ind, ind)
        arr = arr[:, final_ind]
    return arr


def collect(seq, func, kwargs=None, with_index=False, index_col_name=None):
    """
    Take a sequence seq of arguments to function func.
      - func should return a Table or an ndarray.
      - kwargs is a dictionary of arguments that will be passed to func.

    Return the outputs of func concatenated vertically into an np.array
    (thereby making copies of the collected data).
    If the outputs are Tables, concatenate the arrays and return a Table.

    If with_index is True, append index column to outputs.
    If the outputs are Tables, index_col_name must be provided for this purpose.
    """
    all_results = []
    cols = None
    for index, image in enumerate(seq):
        results = func(image, **kwargs) if kwargs else func(image)
        if isinstance(results, Table):
            cols = results.cols
            results = results.arr
        if results.shape[0] > 0:
            if with_index:
                all_results.append(append_index_column(results, index))
            else:
                all_results.append(results)
    ret = np.array([])
    if len(all_results) > 0:
        ret = np.vstack(all_results)
    if cols:
        assert(index_col_name)
        ret = Table(ret, list(cols + [index_col_name]))
    return ret


def collect_with_index(seq, func, kwargs=None, index_col_name=None):
    """
    See collect().
    """
    return collect(seq, func, kwargs, True, index_col_name)


def makedirs(dirname):
    "Does what mkdir -p does, and returns dirname."
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            print("Exception on os.makedirs")
    return dirname


def importance_sample(dist, num_points, kde=None):
    """
    dist is a list of numbers drawn from some distribution.
    If kde is given, uses it, otherwise computes own.
    Return num_points points to sample this dist at, spaced such that
    approximately the same area is between each pair of sample points.
    """
    if not kde:
        kde = st.gaussian_kde(dist.T)
    x = np.linspace(np.min(dist), np.max(dist))
    y = kde.evaluate(x)
    ycum = np.cumsum(y)
    points = np.interp(
        np.linspace(np.min(ycum), np.max(ycum), num_points), xp=ycum, fp=x)
    return points


def fequal(a, b, tol=.0000001):
    """
    Return True if the two floats are very close in value, and False
    otherwise.
    """
    return abs(a - b) < tol


def log2(x):
    "Base-2 log that returns 0 if x==0."
    y = np.atleast_1d(np.copy(x))
    y[y == 0] = 1
    return np.log2(y)


def add_polynomial_terms(X):
    """
    Return matrix X augmented with columns representing 2-dgree polynomial
    expansions of its columns.

    Examples
    --------
    >>> X = np.array([[1,2,3,4],[5,6,7,8]]).T
    array([[1, 5],
           [2, 6],
           [3, 7],
           [4, 8]])
    >>> add_polynomial_terms(X)
    array([[ 1,  5,  1,  5, 25],
           [ 2,  6,  4, 12, 36],
           [ 3,  7,  9, 21, 49],
           [ 4,  8, 16, 32, 64]])

    """
    from itertools import combinations_with_replacement as cwr
    # TODO: will need to combine column indices
    col_combinations = [x for x in cwr(range(X.shape[1]), 2)]
    poly_cols = [np.prod(X[:, c], axis=1) for c in col_combinations]
    return np.hstack((X, np.array(poly_cols).T))


def cartesian(arrays, out=None):
    """
    from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def run_matlab_script(matlab_script_dir, function_string):
    """
    Takes a directory where the desired script is, changes dir to it, runs it with the given function and parameter string, and then chdirs back to where we were.
    """
    if not os.path.exists(matlab_script_dir):
        raise IOError("Cannot find the matlab_script_dir, not doing anything")
    cwd = os.getcwd()
    os.chdir(matlab_script_dir)
    cmd = "matlab -nodesktop -nosplash -r \"%s; exit\"; stty echo" % function_string
    run_command(cmd, loud=True)
    os.chdir(cwd)


def run_command(command, loud=True):
    """
    Runs the passed string as a shell command. If loud, outputs the command and the times. If say exists, outputs it as well. Returns the retcode of the shell command.
    """
    retcode = -1
    if loud:
        print >>sys.stdout, "%s: Running command %s" % (curtime(), command)
        time_start = time.time()

    try:
        retcode = subprocess.call(command, shell=True, executable="/bin/bash")
        if retcode < 0:
            print >>sys.stderr, "Child was terminated by signal ", -retcode
        else:
            print >>sys.stderr, "Child returned ", retcode
    except OSError, e:
        print >>sys.stderr, "%s: Execution failed: " % curtime(), e

    if loud:
        print >>sys.stdout, "%s: Finished running command. Elapsed time: %f" % (curtime(), (time.time() - time_start))
    return retcode


def curtime():
    return time.strftime("%c %Z")
