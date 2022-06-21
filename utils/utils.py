from pathlib import Path
from os import makedirs, path
import numpy as np
from tqdm.auto import tqdm

def make_cache(fpath_in):
    fpath = Path(fpath_in)
    cache_path = path.join(fpath.parent, "cache")
    makdeirs(cache_path)

def progress(what, i, N):
    """Update on progress"""
    print(what + " : " + str(int(i)) + "/" + str(int(N)) + ".\\n")

class TQDM(tqdm):
    """ Subclass of tqdm for progress bars

    https://stackoverflow.com/a/63553373/6877443
    """
    def __init__(self, total, msg = "Progress", kwargs = {}):
        tqdm_params = { 'unit': 'blocks',
                        'unit_scale': True,
                        'leave': False,
                        'miniters': 1,
                        'total': total,
                        'desc': msg}
        tqdm_params.update(kwargs)
        super(TQDM, self).__init__(total, **tqdm_params)



tqdm_params = { 'unit': 'blocks',
                'unit_scale': True,
                'leave': False,
                'miniters': 1
                }
