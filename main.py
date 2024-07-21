import pandas as pd
from glob import glob
from fio import tiff
import numpy as np
from os import path as pth
import re
from itertools import groupby
try:
    from pygbox.utils.SubArgParser import SubArgParser
    from pygbox import namespace as ns
    from pygbox.pygbox import StackContainer, Stack
    from pygbox.fio.name import make_fpath, get_fpath
except ModuleNotFoundError as e:
    from utils.SubArgParser import SubArgParser
    import namespace as ns
    from pygbox import StackContainer, Stack
    from fio.name import make_fpath, get_fpath

def pipeline(args):
    print(args)

    #get paths
    allfpaths = glob(ns.JACOB +
                     r'\2024_06_03 data mining_expansion' +
                     r'\A020_cutouts_oneplane_t_as_is_rois\*_roi.tif')
    allfnames = [pth.split(f)[-1] for f in allfpaths]
    allidxs = [int(re.search("^[0-9]{1,3}?(?=_)", f).group(0)) for f in allfnames]

    # sort
    allidxs, allfnames, allfpaths = zip(*sorted(zip(allidxs, allfnames, allfpaths)))

    #group on idx
    allinfo = list(zip(allidxs, allfnames, allfpaths))
    groupedinfo = [list(v) for k,v in groupby(allinfo, key=lambda x: x[0])]
    groupedfpaths = [[g[-1] for g in gi] for gi in groupedinfo]

    pix2um = (.118, .118, 1.0)
    processedid = 0

    for i, fpaths in enumerate(groupedfpaths):
        ims = []
        # Load planes from valid paths
        for j, fpath in enumerate(fpaths):
            im = tiff.load(fpath, order = "ZT")
            ims.append(im)
        if len(ims) > 1: ims = [np.squeeze(np.stack(ims, 2))] # stack images in Z

        # create Stack object
        try:
            sc = Stack(im = ims[0], fpath = fpath, pix2um = pix2um)
        except IndexError as e:
            import pdb; pdb.set_trace() # figure out what went wrong

        # proces stack
        scc = StackContainer(fpaths = [sc], pix2um = pix2um)
        scc.process({
                'Detector': {
                    'radius': 15, # in pixels
                    'rel_intensity_threshold': 1.3,
                    'rel_min_distance': 2,
                    'objects': get_fpath(fpath, "+_Detector*", ".*"),
                    'verbose': True,
                },
                'Segmentor': {
                    'ext': 30, # in um
                    'corners': get_fpath(fpath, "+_Segmentor*", ".*"),
                    'verbose': True,
                    },
                'Quantifier': {
                    'results': get_fpath(fpath, "+_Quantifier", ".*", {}),
                    'verbose': False,
                    }
        })
        processedid += 1
        #if processedid > 1: break

if __name__ == "__main__":
    ap = SubArgParser(prog="GBOX Analysis", add_help=True)
    args = ap.parse_args()
    pipeline(args)
