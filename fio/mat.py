from scipy.io import loadmat
import h5py
import numpy as np

def load(fname):
    """
    Read v7.3 matlab .mat files
    """
    is_in_list = 1
    if not isinstance(fname, (list, tuple)):
        fname = [fname]
        is_in_list = 0

    results = []
    for f in fname:
        results.append(_load(f))

    if len(fname) == 1 & is_in_list == 0:
        return results[0]
    else:
        return results


def _load(filename):
    """Read v7.3 matlab .mat file

    https://stackoverflow.com/a/58026181
    """

    def conv(path = ''):
        p = path or '/'
        paths[p] = ret = {}
        for k, v in f[p].items():
            if type(v).__name__ == 'Group':
                ret[k] = conv(f'{path}/{k}')  # Nested struct
                continue
            v = v[()]  # It's a Numpy array now
            if v.dtype == 'object':
                # HDF5ObjectReferences are converted into a list of actual pointers
                ret[k] = [r and paths.get(f[r].name, f[r].name) for r in v.flat]
            else:
                # Matrices and other numeric arrays,
                # order arr[..., page, row, col]
                ret[k] = v if v.ndim < 2 else v.swapaxes(-1, -2)
        return ret

    paths = {}
    with h5py.File(filename, 'r') as f:
        return conv()


def drop_flagged(res):
    """
    walk over results and check if flagged as bad
    """

    if not isinstance(res, (list, tuple)): res = [res]
    good_res = []

    for r in res:
        if r['iStat']['flagged'] == 0:
            good_res.append(r)
    return good_res


def parse_fields(x, key):
    """parse fields for given (nested) (set of) key(s)"""
    
    if not isinstance(key, (list, tuple)): key = [key]
    if len(key) == 1:
        return  [y[key[0]] for y in x]

    xout = []
    temp = x
    for kk in key:
        try:
            temp = [y[kk] for y in temp]
        except TypeError as e:
            for k in kk:
                xout.append(np.asarray([y[k] for y in temp]))
    #return np.array(xout).flatten()
    return xout
