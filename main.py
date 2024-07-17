import pandas as pd
from glob import glob
from fio import tiff
import numpy as np
try:
    from pygbox.utils.SubArgParser import SubArgParser
    from pygbox import namespace as ns
    from pygbox.pygbox import StackContainer, Stack
    from pygbox.fio.name import make_fpath
except ModuleNotFoundError as e:
    from utils.SubArgParser import SubArgParser
    import namespace as ns
    from pygbox import StackContainer, Stack
    from fio.name import make_fpath

def pipeline(args):
    print(args)

    #read data annotation
    filepath_df = glob(ns.HOME + r'\surfdrive\data\*addition*_no_drift_cor*\*data.csv')
    df = pd.read_csv(filepath_df[0], sep = ";", header = 0)
    # extract file ids
    common_column = "order_of_appearance"
    ids = df[common_column].unique()
    # pull out names
    names = []
    for ix in ids:
        subdf = df[df[common_column] == ix]
        names.append(subdf["name"].unique().tolist())
    #fetch file paths
    allfpaths = []
    for nm in names:
        fpaths = [glob(ns.HOME + r'\\Downloads\\data\\A020*_addition\\' + x + '.tif') for x in nm]
        allfpaths.append(fpaths)

    pix2um = (.118, .118, 1.0)
    processedid = 0

    for i, fpaths in enumerate(allfpaths):
        ims = []
        # Load planes from valid paths
        for j, fpath in enumerate(fpaths):
            if len(fpath) == 0: continue
            im = tiff.load(fpath, order = "ZT")
            ims.append(im)
        if len(ims) == 0: continue
        if len(ims) > 1: ims = [np.squeeze(np.stack(ims, 2))] # stack images in Z

        # create Stack object
        try:
            sc = Stack(im = ims[0][..., 0:10], fpath = fpath[0], pix2um = pix2um)
        except IndexError as e:
            import pdb; pdb.set_trace() # figure out what went wrong

        scc = StackContainer(fpaths = [sc], pix2um = pix2um)
        scc.process({
                'Detector': {
                    'radius': 15, # in pixels
                    'rel_intensity_threshold': 1.5,
                    'rel_min_distance': 8,
                    #'objects': glob(make_fpath(fpath[0], "+_Detector*", ".*", append_date = False)).pop(),
                    'objects': [],
                    'verbose': False,
                },
                'Segmentor': {
                    'ext': 30, # in um
                    'corners': glob(make_fpath(fpath[0], "+_Segmentor*", ".*", append_date = False)).pop(),
                    #'corners': [],
                    'verbose': True,
                    },
                'Quantifier': {
                    #'results': glob(make_fpath(f, "+_Quantifier*", ".*", append_date = False)).pop(),
                    'results': {},
                    'verbose': False,
                    }
        })
        processedid += 1
        #if processedid > 1: break

if __name__ == "__main__":
    ap = SubArgParser(prog="GBOX Analysis", add_help=True)
    args = ap.parse_args()
    pipeline(args)
