from os import path
import tifffile
import numpy as np
from datetime import datetime

class Image(object):
    def __init__(self, fpath):
        self.direc, self.fname, self.ext = self._parse_fpath(fpath)
        self.impath = fpath

    def _parse_fpath(self, fpath):
        assert path.isfile(fpath), 'File does not exist'

        parent, base = path.split(fpath)
        base, ext = path.splitext(base)
        return parent, base, ext

    def load(self):
        self.im = load(self.impath)

    def reshape(self):
        self.im = reshape(self.im, self.nZ, self.nT)

    def save(self, im = None):
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = path.join(self.direc, self.fname + "_" + dt + self.ext)
        im = self.im if not im else im
        save(im, fpath)


def reshape(ims, nZ = 1, nT = 1):
    """
    reshape images to expected 3rd and 4th dimension
    """
    if not isinstance(ims, (list, tuple)): ims = [ims]
    for i,im in enumerate(ims):
        im = np.squeeze(im)
        im = np.split(im, nT, -1)
        im = np.stack(im, im[0].ndim)
        ims[i] = im
    ims = ims[0] if len(ims) == 1 else ims
    return ims

def load(impath):
    """
    loads tiff from impath
    It reads in the order TZYX
    """
    if not isinstance(impath, (list, tuple)): impath = [impath]
    imstacks = []
    for imp in impath:
        assert path.isfile(imp), "Supply correct image path."
        with tifffile.TiffFile(imp) as tif:
            imstack = tif.asarray()
            #imstack = np.swapaxes(imstack, -1, -2) # ..YX to ..XY
            imstack = np.moveaxis(imstack, 0, -1) # ..XY to XY..
            # make sure image has always four dimensions
            shape = imstack.shape
            shape = shape + (1, ) * (4 - len(shape))
            imstack = np.reshape(imstack, shape)
            imstacks.append(imstack)
    imstacks = imstacks[0] if len(imstacks) == 1 else imstacks
    return imstacks

def save(im, impath = 'test.tif'):
    """
    assumes the data is in XY.. shape
    """
    if im.ndim < 4: im = np.expand_dims(im, -1)
    opts = {'contiguous': True, 'metadata': {'axes': 'XYCZT'}}
    nonxy_dims = im.shape[2:]
    with tifffile.TiffWriter(impath, imagej = True) as tif:
        for j in range(nonxy_dims[-1]):
            for i in range(nonxy_dims[-2]):
                im_ = im[:,:,i, j]
                try:
                    tif.write(im_, **opts)
                except AttributeError as e:
                    tif.save(im_, **opts)

def get_tiff_metadata(tif, tiftype = "imagej"):
    """
    Obtain metadata from image
    """
    assert isinstance(tiftype, (str, )), "Tiftype must be a string."
    tiftype = tiftype.lower()
    assert tiftype in ["imagej", "andor"], "Unrecognized tiftype."
    command = tiftype + "_metadata"

    mdata_out = {}
    mdata_in = getattr(tif, command)

    pairs = (
        ("numChannels", "channels"),
        ("numSlices", "slices"),
        ("numT", "frames"),
        ("numPos", "positions")
    )
    for p in pairs:
        try:
            mdata_out[p[0]] = mdata_in[p[1]]
        except KeyError as e:
            mdata_out[p[0]] = 1

    return mdata_out
