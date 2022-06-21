from os import path
from glob import glob
import numpy as np
import csv
import pandas as pd

def load_tracks(fpath, ftype = 'csv', usecols = None, skiprows = None, kwargs = {}):
    """load tracks"""

    ftype = ftype.lstrip('.').lower()

    if ftype == 'csv':
        #with open(fpath, 'r') as f:
        #    for row in csv.reader(f):
        #        pass
        tracks = pd.read_csv(fpath, usecols = usecols, skiprows = skiprows, **kwargs)
    elif ftype == 'xml':
        raise NotImplementedError("Update to pandas 1.3+ to read xml!")
        tracks = pd.read_xml(fpath)
    elif ftype == 'tsv':
        tracks = pd.read_csv(fpath, delimiter = '\t', header = None,
                        usecols = usecols, skiprows = None,
                        names = ['T', 'X', 'Y'], **kwargs)
    else:
        raise NotImplementedError('Filetype not supported!')
    return tracks
