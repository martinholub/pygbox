from os import path as pth
from scipy.ndimage import median_filter, gaussian_laplace
from skimage.measure import regionprops, label
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
from itertools import groupby
from copy import deepcopy

try:
    from pygbox.fio.meta import load_metadata as loadmeta
    from pygbox.fio.meta import dump_json, load_json
    from pygbox.fio import tiff
    from pygbox.utils.utils import TQDM, tqdm_params
    from pygbox import ops, masking
    from pygbox.viz import viz, plots
    from pygbox import detection
except ModuleNotFoundError as e:
    from fio.meta import load_metadata as loadmeta
    from fio.meta import dump_json, load_json
    from fio import tiff
    from utils.utils import TQDM, tqdm_params
    from viz import viz, plots
    import ops, masking, detection


# TODO: uneven illumination correction

# importing submodules: https://stackoverflow.com/a/8899345/6877443

class StackContainer(object):
    """ Container Object for stacks

    Examples:
        ```
        # Will quickly run out of memory
        sc = StackContainer(fpaths, pix2um)
        sc.load()
        sc.detect()
        sc.segment()
        sc.quantify()
        rg = sc.parse_result()

        # More efficient for memory
        sc = StackContainer(fpaths, pix2um)
        sc.process()
        sc.parse_results()
        ```
    """
    def __init__(self, fpaths = [""], pix2um = (1, 1, 1)):
        self.fpaths = fpaths
        self.pix2um = pix2um

    @property
    def pix2um(self):
        return self._pix2um

    @pix2um.setter
    def pix2um(self, value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            assert value is not None, "Pixel size must be a finite number!"
            value = [value] * 3
        self._pix2um = np.asarray(value)

    def load(self):
        raise NotImplementedError("Please process stacks sequentially with `process` method!")
        self.stacks = []
        for f in self.fpaths:
            s = Stack(fpath = f, pix2um = self.pix2um)
            s.load()
            self.stacks.append(s)

    def detect(self):
        raise NotImplementedError("Please process stacks sequentially with `process` method!")
        self.detectors = []
        for s in self.stacks:
            d = Detector()
            d.detect(s, pix2um = self.pix2um)
            self.detectors.append(d)

    def segment(self, ext = 25):
        raise NotImplementedError("Please process stacks sequentially with `process` method!")
        self.segmentors = []
        for s, d in zip(self.stacks, self.detectors):
            sg = Segmentor(s, d)
            sg.segment(ext)
            self.segmentors.append(sg)
    def quantify(self):
        raise NotImplementedError("Please process stacks sequentially with `process` method!")
        for sg in self.segmentors:
            q = Quantifier(sg)
            q.quantify()
            self.quantifiers.append(q)

    def process(self, kwargs = {}):
        """ Process stacks sequentially """

        kwargs_ = {'Detector': {}, 'Segmentor': {'ext': 25}, 'Quantifier': {}}
        kwargs_.update(kwargs)

        self.results = {}

        print("Processing {:d} files.".format(len(self.fpaths)))

        tqdm_params_ = {'total': len(self.fpaths),
                        'desc' : type(self).__name__ + " processing:"}
        tqdm_params.update(tqdm_params_)

        with tqdm(**tqdm_params) as pbar: # progbar
            for i, f in enumerate(self.fpaths):
                pbar.update(1)

                if isinstance(f, (Stack, )):
                    s = f
                else:
                    fname = pth.splitext(pth.split(f)[-1])[0]
                    s = Stack(fpath = f, pix2um = self.pix2um)
                    s.load()

                d = Detector(**kwargs_['Detector'])
                d.detect(s, pix2um = self.pix2um)

                sg = Segmentor(s, d, **kwargs_['Segmentor'])
                sg.segment()

                q = Quantifier(sg, **kwargs_['Quantifier'])
                q.quantify()

                self.results[str(i)] = q.results
                del s, d, sg, q
                #return d, sg, q


    def parse_result(self, kwargs = {}):
        #raise NotImplementedError("Please call process stacks sequentially with `process` method!")
        out = []
        try:
            for i, q in enumerate(self.quantifiers):
                out.extend(q.parse_result(**kwargs))
        except AttributeError as e:
            out = self._parse_result(**kwargs)
        out = np.asarray(out)
        return out[~np.isnan(out)]

    def _parse_result(self, **kwargs):
        """ Parse result of quantifier
        """
        out = []
        for _,v in self.results.items():
            q = Quantifier() # could do it also without quantifier, but then cannot use its fun
            q.results = v
            out.extend(q.parse_result(**kwargs))
            del q
        return out


    def save(self, fpath = ""):
        if not fpath:
            fpath = self.fpaths
            if isinstance(fpath, (list, tuple)):
                if len(fpath) > 1:
                    fpath = pth.split(fpath[0])[0]
                else:
                    fpath = fpath[0]
        else:
            fpath_ = self.fpaths
            if not pth.isabs(fpath):
                fpath_ = pth.split(fpath_[0])[0]
                fpath = pth.join(fpath_, fpath)

        fpath = fpath + "_" + type(self).__name__
        dump_json(self.results, fpath)

    def load(self, fpath = ""):
        if not fpath:
            fpath = self.fpath

        fpath = pth.split(fpath)[0] + "\\"
        fpath = glob(fpath + "*" + type(self).__name__ + "*.json")[0]
        self.result = load_json(fpath)

class Stack(object):
    """Image Stack

    Args:
        im (np.ndarray): image data, if not supplied `fpath` is required
        fpath (str): path to image data
        pix2um (array_like): pixel size for all three dimensions
            if single value supplied, this is used for all dimension


    Examples:
        ```
        s = Stack(fpath = impath, pix2um = 1)
        s.load()
        ```

    """

    def __init__(self, im = np.empty(0), fpath = '', pix2um = (1, 1, 1)):
        self.im = im
        self.fpath = fpath
        self.pix2um = pix2um

    @property
    def pix2um(self):
        return self._pix2um

    @pix2um.setter
    def pix2um(self, value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            assert value is not None, "Pixel size must be a finite number!"
            value = [value] * 3
        self._pix2um = np.asarray(value)


    def load(self, fpath = None):
        """ Load stack(s) from filepath
        """
        # TODO: how to handle multiple stacks in one path?
        if not fpath:
            fpath = self.fpath
        else:
            self.fpath = fpath
        if not pth.isfile(fpath):
            print(fpath + ": Does not exist and will be created")
        self.im = tiff.load(fpath)

    def load_metadata(self, fname = ''):
        """ Load metadata from txt file
        """
        if fname: #filename was passed
            fpath = pth.splitext(self.fpath)[0]
            fpath = pth.join(fpath, fname) + ".txt"
        else: # expect same filename as the image file
            fpath = self.fpath
        try:
            metadata = loadmeta(fpath)
        except Exception as e:
            try: # try to fetch any txt file
                fpath = pth.split(fpath)[0]
                fpath = glob(pth.join(fpath, '*.txt'))[0]
                metadata = loadmeta(fpath)
            except Exception as e:
                metadata = {}
        self.metadata = metadata

    def describe(self):
        """ Compute image descriptive statistics """
        self.max = np.max(self.im)
        self.min = np.min(self.im)
        self.median = np.median(self.im)
        self.mean = np.mean(self.im)
        (self.nX, self.nY, self.nZ, self.nT) = self.im.shape

    def detect(self, kwargs = {}):
        """ Object detection
        """
        raise NotImplementedError("Please call Detector directly!")

        self.Detector = Detector(**kwargs)
        # results in self.Detector.objects
        self.Detector.detect(self.im)

    def segment(self, kwargs = {}):
        """ Segmentation: separate bg and fg by masking
        """
        raise NotImplementedError("Please call Segmentor directly!")
        self.Segmentor = Segmentor(**kwargs)
        # results in self.Segmentor.masks
        self.Segmentor.segment(self.im, self.Detector.objects)

class Detector(object):
    """ Object Detector

    Args:
        radius (int): expected object radius
        rel_intensity_threshold (float): relative threshold for keeping bright objects
            TODO: use threshold from model???
        rel_min_distance (float): minimal relative distance between spots
            spots closer than the relative distance are averaged
        verbose (bool): turn on verbose messages

    Examples:
        ```
        s = Stack(im)
        d = Detector(s)
        d.detect()
        ```
    """
    def __init__(   self, radius = 20, rel_intensity_threshold = 2.5,
                    rel_min_distance = 2.5, verbose = False, objects = []):
        # attribs to be populated
        self.objects = objects

        # attribs passed
        self.radius = int(np.round(radius))
        self.rel_intensity_threshold = rel_intensity_threshold
        self.rel_min_distance = rel_min_distance
        self.verbose = verbose


    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        if isinstance(value, (str, )):
            path = Path(value)
            if path.is_file():
                value = path
            else:
                print("Detector.objects: Path is not a file.")
                value = []
        self._objects = value

    def __str__(self):
        return self.__class__.__name__

    def detect(self, im, pix2um, fpath = None):
        """ Detect objects in a stack """

        # Load object
        do_save = 1
        if self.objects:
            fpath = Path(self.objects)
            if fpath.is_file():
                try:
                    self.load(fpath) # assings self.objects
                    do_save = 0
                    if self.verbose:
                        objects = detection.inspect_spots(im.im, self.objects, self.radius, pix2um)
                        if len(self.objects) == len(objects):
                            if np.all([(x == y).all() for x,y in zip(self.objects, objects)]):
                                pass
                            else:
                                self.objects = objects; do_save = 1
                        else:
                            self.objects = objects; do_save = 1

                    if do_save:
                        self.save(fpath) # save changes if any made
                        do_save = 0
                    return
                except Exception as e:
                    do_save = 1
                    print(f"{type(e).__name__} was raised: {e}")

        try: # check if instance of Stack is passed in
            fpath = im.fpath
            im = im.im
        except AttributeError as e:
            assert fpath is not None, "Detector.detect must supply correct fpath"

        # scaling factor for sum
        try:
            scaler = np.iinfo(im.dtype).max
        except ValueError as e:
            scaler = np.finfo(im.dtype).max

        nX, nY, nZ, nT = im.shape
        objects = []
        for t in range(0, nT):
            if im.shape[2] == 1:
                xspot, yspot, zspot = detection.detect2D(im[:, :, 0, t],
                    self.radius, self.rel_intensity_threshold,
                    self.rel_min_distance, projection = 'max')

            else:
                xspot, yspot, zspot = detection.detect3D(
                    im[:, :, :, t], self.radius, pix2um, opt = 2,
                    rel_min_distance = self.rel_min_distance,
                    rel_intensity_threshold = self.rel_intensity_threshold)

            #if len(xspot) == 0: xspot, yspot, zspot = (np.asarray([np.nan]), )*3
            coords = np.array((xspot, yspot, zspot, np.ones_like(zspot, dtype = np.int)*t)).T
            objects.append(coords)

        objects = [o for o in objects if len(o) > 0]
        if self.verbose:
        # manually confirm spots
            objects = detection.inspect_spots(im, objects, self.radius, pix2um)

        self.objects = objects
        if do_save:
            self.save(fpath)

    def save(self, fpath = ""):
        if not fpath:
            fpath = make_fpath(
                self.stack.fpath, "+_" + type(self).__name__, ".npy",
                append_date = False)
        else:
            fpath = pth.splitext(fpath)[0]
            if not type(self).__name__ in fpath:
                fpath = fpath + "_" + type(self).__name__
        np.save(fpath, self.objects)

    def load(self, fpath = ""):
        if not fpath:
            fpath = self.stack.im.fpath
            fpath = pth.splitext(fpath)[0]
            fpath = glob(fpath + "*" + type(self).__name__ + "*")[0]
        objects = np.load(fpath, allow_pickle = True).tolist()
        #self.objects = [x.tolist() for x in objects] # lists of len [N] for each [t]
        self.objects = objects #MH240705


class Tracker(object):
    """ Tracks objects through time
    """
    def __init__(self):
        pass

class Segmentor(object):
    """ Background and foreground segmentor

    Args:
        stack: instance of pygbox.Stack
        detector: detector pygbox.Detector()

    Examples:
        ```
        s = Stack(im)
        d = Detector(s)
        d.detect()
        sg = Segmentor(s, d)
        sg.segment()
        # plot (sg.fetch_crop(), sg.fetch_mask())
        ```

    """
    def __init__(   self, stack = Stack(), detector = Detector(), ext = 20,
                    corners = [], verbose = None):

        # Attribs to be populated
        #self.masks = []
        #self.crops = []
        self.corners = corners
        self.ext = ext

        # Attribs passed
        self.stack = stack
        self.detector = detector
        self.verbose = self.detector.verbose if verbose is None else verbose

        @property
        def corners(self):
            return self._corners

        @corners.setter
        def corners(self, value):
            if isinstance(value, (str, )):
                path = Path(value)
                if path.is_file():
                    value = path
                else:
                    print("Segmentor.corners: Path is not a file.")
                    value = []
            self._corners = value


    def segment(self):
        """ Create mask separating background from foreground

        Args:
            ext (int): size crop within which to segment, in micrometers

        Assigns:
            masks (list): per object per time nested list of masks
            corners (list): per object per time nested list of crop corners
        """

        im = self.stack.im
        nX, nY, nZ, nT = im.shape
        # select extent to capture X um in real world coordinates
        if self.stack.pix2um[-1] > 0:
            ext = np.round( self.ext / self.stack.pix2um ).astype(int)
            if ext[2] % 2: ext[2] += 1
        else:
            ext = np.round( self.ext / self.stack.pix2um[:2] ).astype(int)
            ext = np.append(ext, 1)
        if ext[0] % 2: ext[0] += 1
        if ext[1] % 2: ext[1] += 1

        # Load object
        if self.corners:
            fpath = Path(self.corners)
            if fpath.is_file():
                try:
                    self.load(fpath) # assings self.corners, self.mask
                    if self.verbose:

                        new_mask = masking.inspect_mask(im, self.mask, self.stack.pix2um)
                        if not (new_mask.sum() == self.mask.sum()):
                            self.mask = new_mask
                            self.save(fpath)
                    return
                except Exception as e:
                    print(f"{type(e).__name__} was raised: {e}")
                    pass

        # Fetch data to analyze
        centrs = self.detector.objects
        # sort and group found spots
        idx = [o[-1] for o in centrs]
        #idx, centrs = zip(*sorted(zip(idx, centrs)))
        centrs = sorted(centrs, key = lambda x: x[-1])
        #np.where(~np.isin(np.arange(100), idx)) # could do this
        centrs = [list(v) for k, v in groupby(centrs, key = lambda x: x[-1])]
        # loop over object centroids
        masks_all_list = []; ims_all = []; spans_all = []; threshs_all = []; snrs_all = [];
        nX, nY, nZ, nT = im.shape
        masks_all = np.zeros((nX, nY, nZ, nT), dtype = np.bool)

        for itc, tc in enumerate(centrs): # loop over time-points
            masks = []; ims = []; spans = []; threshs = []; snrs = []
            for ic, c_ in enumerate(tc): # loop over centroids
                t = int(c_[-1])
                #MH 240719 - at the moment works only with 1 spot per frame
                # crop out object of interest
                xspan, yspan, zspan = masking.get_span_axs(ext, c_, (nX, nY, nZ))
                xspan, yspan, zspan = zip((0, 0, 0), (nX, nY, nZ)) # MH
                #xspan = [np.max([c_[0] - ext[0]//2, 0]), np.min([nX, c_[0] + ext[0]//2])]
                #yspan = [np.max([c_[1] - ext[1]//2, 0]), np.min([nY, c_[1] + ext[1]//2])]
                #zspan = [np.max([c_[2] - ext[2]//2, 0]), np.min([nZ, c_[2] + ext[2]//2])]
                im_ = im[xspan[0]:xspan[1], yspan[0]:yspan[1], zspan[0]:zspan[1], t]
                #TODO: make the check below relative to pixel_size
                # if any([x <= 6 for x in im_.shape[:np.sum(self.stack.pix2um > 0)]]):
                #     mask = np.zeros_like(im_)
                #     continue

                # Mercator mask
                # mask = masking.mercator_mask(im_, self.pix2um)
                # thresh = 0; snr = np.nan

                # Plane by plane mask
                mask = masking.planar_mask(im_, self.stack.pix2um)

                #radial mask
                #mask = masking.radial_mask(im_, self.detector.radius, self.stack.pix2um)

                #apply some 3D checks on the mask
                #if not masking.qc_mask3D(mask): continue

                # store info
                masks.append(mask); spans.append((xspan, yspan, zspan)); #ims.append(im_)
                #threshs.append(thresh); snrs.append(snr)

            # store info at highest level
            if len(masks) == 0:
                masks = [np.zeros((nX, nY, nZ), dtype = np.bool)]
            elif len(masks) > 1:
                #raise NotImplementedError
                # MH240723 though this would work, can try later
                mask = np.zeros_like(masks[0],  dtype = np.int)
                for i, mm in enumerate(masks):
                    mm = mm.astype(np.int)
                    mm[mm>0] = i+1 #asign label in thos mask
                    mask += mm #assign label in mask of masks
                masks = [mask]

            masks_all_list.append(masks); spans_all.append(spans); # ims_all.append(ims)
            masks_all[:, :, :, t] = masks[0]
            #threshs_all.append(threshs); snrs_all.append(snrs)

        self.corners = spans_all; #self.masks = masks_all; # self.crops = ims_all
        #self.threshs = threshs_all; self.snrs = snrs_all
        #masks_all = self.register_masks(masks_all) #MH240704

        # should work also with a single mask per frame
        #masks_all = np.stack([x[0] for x in masks_all], -1)

        # asign masks value across time using some smart heuristic
        #masks_all[..., :] = np.quantile(masks_all.astype(int),
        #                                q = .5, axis = -1, keepdims = True) > .5

        # force mask atthe top and bottom layers to be zero if mask is thick
        if masks_all.shape[2] > 5:
            masks_all[:, :, 0, :] = 0; masks_all[:, :, -1, :] = 0;

        if self.verbose:
            new_mask = masking.inspect_mask(im, masks_all, self.stack.pix2um)
            if not (new_mask.sum() == masks_all.sum()):
                self.mask = new_mask
            else:
                self.mask = masks_all
        self.save()

    def register_masks(self, masks = None):
        """ Assign pixels uniquely to objects within a mask

        Walk through masks and:
            1) register them in larger FOV
            2) decide if voxels are overlapping between masks
            3) label all voxels uniquely based on some heuristic

        Args:
            masks (list of ndarray): list of 3D boolean masks
        Requires:
            self.stack.im, self.corners

        Notes:
            - Probably simplify by working more on FOV, instead of crop.
        """
        # Mask of the size of whole FOV
        mask = np.zeros(self.stack.im.shape, dtype = np.int)
        if not masks: masks = self.mask # can also pass as argument if desired

        # store info on overlapping masks
        lab_overlaps = []

        # first loop over time:
        for tt in range(mask.shape[3]):
            label = 0 # unique label


            # then loop over objects:
            for jj, (fgm, crn) in enumerate(zip(masks[tt], self.corners[tt])):
                label = jj+1 # start from 1

                fgm = fgm.astype(np.int) # allow assignment of nonboolean
                fgm[fgm == True] = label # set mask value to unqiue label

                # fetch crop full FOV mask
                mask_crop = mask[   crn[0][0]:crn[0][1],
                                    crn[1][0]:crn[1][1],
                                    crn[2][0]:crn[2][1], tt]

                # Are there already ovelrapping labels in the crop?
                labels = np.unique(mask_crop[fgm > 0])
                labels = np.delete(labels, labels == 0) # 0 is bakcground, drop it

                if len(labels) > 0: # if crop already contains other labels
                    # find indices of all already labeled voxels
                    idxs0 = [np.flatnonzero(mask_crop == il) for il in labels]
                    # find indices that are to be newly labeled
                    idx1 = np.flatnonzero(fgm)

                    # intersect these indices
                    intersect_ids = []
                    for l0, idx in zip(labels, idxs0):
                        intersect_id_ = idx1[np.isin(idx1, idx)]
                        # convert index to 3D (bit clumsy, but works)
                        intersect_id = np.unravel_index(intersect_id_, mask_crop.shape)

                        # 0: partial overlap, 1: new is subset of old, 2: old is subset of new
                        # Approaches A/B - not clear what will work better on long run
                        case = 0
                        if len(intersect_id_) == len(idx1):
                            case = 1 if (intersect_id_ == idx1).all() else 0
                        elif len(intersect_id_) == len(idx):
                            case = 2 if (intersect_id_ == idx).all() else 0

                        if case == 1:
                            #A: the new label is a susbet of an exisitng label, assign the new one
                            #mask_crop[intersect_id[0], intersect_id[1], intersect_id[2]] = label

                            #B: the new label is a susbet of an exisitng label, dont assign it
                            # no code needed

                            intersect_id = np.array([])
                        elif case == 2:
                            # A: the new label entirely ovewrites an existing label, preserve the old one
                            # no code needed

                            #B: the new label entirely ovewrites an existing label, use it
                            mask_crop[intersect_id[0], intersect_id[1], intersect_id[2]] = label
                            intersect_id = np.array([])
                        elif case == 0:
                            # assign the more prominent label
                            label = label if len(idx1) > len(idx) else l0
                            fgm[fgm > 0] = label

                            # the masks are overlapping, assign value of choice
                            mask_crop[intersect_id[0], intersect_id[1], intersect_id[2]] = label
                            # register index to full FOV mask
                            intersect_id = np.stack(intersect_id, axis = 1)
                            intersect_id[:, 0] += crn[0][0]
                            intersect_id[:, 1] += crn[1][0]
                            intersect_id[:, 2] += crn[2][0]
                        intersect_ids.append(intersect_id)
                    # store information on overlaps
                    lab_overlaps.append((label, labels.tolist(), intersect_ids))

                # Asign label if still needed, but only where you can do it uniquely
                mask_crop[mask_crop == 0] = fgm[mask_crop == 0]
                mask[   crn[0][0]:crn[0][1],
                        crn[1][0]:crn[1][1],
                        crn[2][0]:crn[2][1], tt] = mask_crop

            # Walk through overlap voxels, and assign to object based on heuristic
            regprops = regionprops(mask[..., tt])
            reglabs = [r.label for r in regprops]
            for l1, l0s, idx0s in lab_overlaps:
                if l1 not in reglabs: continue # was overwritten by other overlaps
                for l0, idx in zip(l0s, idx0s):
                    # skip heuristic-based assignment if assigned earlier manually
                    if idx.size == 0: continue
                    if l0 not in reglabs: continue # was overwritten by other overlaps

                    # distance based asignment
                    # TODO: account for axis assymetry??
                    #cm1 = regprops[reglabs.index(l1)].centroid # center of mass of label
                    #cm0 = regprops[reglabs.index(l0)].centroid # center of mass of competitor
                    #d1 = np.sum((idx - cm1)**2, axis = 1) # sq.dist. of voxels from CM of label
                    #d0 = np.sum((idx - cm0)**2, axis = 1) # sq.dist. of voxels from CM of competitor

                    # size based asignment
                    try:
                        d1 = 1/regprops[reglabs.index(l1)].area
                        d0 = 1/regprops[reglabs.index(l0)].area
                        # decide which label is closer
                        labs = np.where(d1 <= d0, l1, l0)
                    except ValueError as e: # one of the indices is not in reglabs?
                        raise NotImplementedError("Either of ({}, {} indices not in labels.)".format(l1, l0))
                    mask[idx[:, 0], idx[:, 1], idx[:, 2], tt] = labs


        # walk through labels, and clea up the mask
        regprops = regionprops(mask[..., tt])
        bad_labels = []
        for r in regprops:
            is_good = 1

            # check size of the area
            #vol_expect = 3/4 * np.pi * np.prod(1/self.stack.pix2um)
            #if (r.area < vol_expect * 0.1) or (r.area > 10*vol_expect):
            #    is_good = 0

            #check if the mask is at the edge
            # if (np.in1d([mask.shape[0], 0], r.bbox)).any():
            #     is_good = 0
            # if (mask.shape[2] <= r.bbox[-1]):
            #     is_good = 0

            # check if mask too thin in any direction
            bbox = np.array(r.bbox)
            if any((bbox[3:] - bbox[:3]) < (1/self.stack.pix2um)):
                is_good = 0

            #if r.solidity < 0.3: # sometimes fails on convexhull
            #    is_good = 0

            #if r.area/r.area_bbox < 0.1: # equal to r.extent
            #    is_good = 0

            # delete the label
            if not is_good:
                mask[mask == r.label] = 0

        return mask

    def fetch_crop(self, icrop = 0, itime = None, im = None, pads = (0, 0, 0, 0)):
        """ Fetch an image crop knowing corners
        """
        # try to get adta from stack
        try:
            im = self.stack.im
        except AttributeError as e:
            pass

        if len(self.corners) == 1:
            axspans = self.corners[0][icrop]
        else:
            axspans = self.corners[itime][icrop]
        # assert len(axspans) == im.ndim, "Axes spans do not match image dimensions".
        if len(axspans) < im.ndim:
            for i, d in enumerate(im.shape):
                if len(axspans) < (i + 1):
                    if isinstance(axspans, (tuple, list, )):
                        axspans += ((0, d), )
                    elif isinstance(axspans, (np.ndarray, )):
                        axspans = np.append(axspans, ((0, d), ), axis = 0)
        # apply crop expansion/shrinking if desired
        axspans = [(x[0] - y, x[1] + y) for x,y in zip(axspans, pads)]

        # assume 4D image by convention
        crop = im[  axspans[0][0] : axspans[0][1], axspans[1][0] : axspans[1][1],
                    axspans[2][0] : axspans[2][1], axspans[3][0] : axspans[3][1]]
        return crop, axspans

    def fetch_mask(self, imask = 0, itime = 0, pads = (0, 0, 0, 0)):
        """ Fetch mask at id `imask` at timepoint `itime`

        TODO:
            Ideally set every label other than the 'imask' equal to 0.
            However, not sure if that would be correct, cuz the ordinal may not be aligned with
            the label number :((( something to fix as time goes on
        """
        try:
            return self.masks[itime][imask], None
        except (AttributeError, IndexError) as e:
            fullmask = self.mask[:, :, :, itime]
            axspans = self.corners[itime][imask]
            #import pdb; pdb.set_trace()
            axspans = [(x[0] - y, x[1] + y) for x,y in zip(axspans, pads)]
            mask = fullmask[axspans[0][0] : axspans[0][1], axspans[1][0] : axspans[1][1],
                            axspans[2][0] : axspans[2][1]]

            if len(np.unique(mask)) > 2: # multiple labels in the crop - happens a lot!
                # pas binary to the fun
                mask = mask.astype(bool) # convert label to {0,1} boolean
                mask = masking.mask_central_object(mask)[0] #could also do it via connected components

            return mask.astype(bool), axspans

    def save(self, fpath = ""):
        if not fpath:
            fpath = self.stack.fpath
        fpath = pth.splitext(fpath)[0]
        if not type(self).__name__ in fpath:
            fpath = fpath + "_" + type(self).__name__
        np.savez(fpath, corners = self.corners, mask = self.mask)

    def load(self, fpath = ""):
        if not fpath:
            fpath = self.stack.fpath
            fpath = pth.splitext(fpath)[0]
            fpath = glob(fpath + "*" + type(self).__name__ + "*")[0]
        segdict = np.load(fpath, allow_pickle = True)
        self.corners = segdict['corners']; self.mask = segdict['mask']

class Quantifier(object):
    """ Quantify properties of object(s)
    """

    def __init__(   self, segmentor = Segmentor(), results = {},
                    verbose = None):
        self.segmentor = segmentor
        self.results = results
        self.verbose = self.segmentor.verbose if verbose is None else verbose

        @property
        def results(self):
            return self._results

        @results.setter
        def results(self, value):
            if isinstance(value, (str, )):
                path = Path(value)
                if path.is_file():
                    value = path
                else:
                    print("Segmentor.corners: Path is not a file.")
                    value = {}
            self._results = value

    def get_threshold(self, im = None):
        """ Calculate BG threshold on raw data

        Previously calculated but on processed data, so do it again here.

        TODO:
            Propagate the thresholding function from `masking` and choose automatically here.

        """
        im_quant, thresh, snr = masking.threshold_model(im)
        return im_quant, thresh, snr

    def quantify(   self, icrop = None, itime = None,
                    descriptors = ('radius_of_gyration', 'sum_intensity'),
                    kwargs = {'do_bg_sub': False}):

        """ Evaluate decriptors on objects.

        Note:
            - Maybe be slow because looping over dict?
        """

        do_assign = 1
        if self.results:
            self.load(self.results)
            results = self.results
            do_assign = 0
            print("Quantifier: Loaded previous results.")
        else:
            if not itime:
                itime = np.arange(self.segmentor.mask.shape[3], dtype = int)
            if isinstance(icrop, (int)):
                icrop = [icrop] * len(itime)
            if not icrop:
                # get unique labels, ignore 0=background
                #icrop = [np.unique(self.segmentor.mask[..., tt])[1:] - 1 for tt in itime]
                # get unique labels, incl 0=background
                if self.segmentor.mask.dtype == np.bool:
                    intmask = self.segmentor.mask.astype(np.int)
                else:
                    intmask = self.segmentor.mask
                icrop = [np.unique(intmask[..., tt]) for tt in itime]
                #icrop = [\
                #    np.unique(self.segmentor.mask[..., tt])[1:] if \
                #    len(np.unique(self.segmentor.mask[..., tt])) > 1 \
                #    else np.array([0]) for tt in itime]

            if not isinstance(descriptors, (tuple, list, )): descriptors = [descriptors]
            if isinstance(descriptors, (tuple, )): descriptors = list(descriptors)

            kwargs.update({'dx': self.segmentor.stack.pix2um})
            results = {str(it):{str(ic):{} for icc in icrop for ic in icc} for it in itime}
            axspans = []
            for it in list(results.keys()):
                axspans_it = []
                for i, ic in enumerate(list(results[str(it)].keys())):
                    if int(ic) == 0: continue
                    if not results[str(it)][str(ic)]: #Seems superfluous

                        # MH240704
                        #fgm_, _ = self.segmentor.fetch_mask(i, int(it))
                        #im_bg, axspan = self.segmentor.fetch_crop(i, int(it))
                        fgm_ = self.segmentor.mask[..., int(it)]
                        im_bg = self.segmentor.stack.im[..., int(it)]

                        axspan = [*zip((0, 0, 0), im_bg.shape + (1, ))]
                        im_, thresh, snr = self.get_threshold(im_bg)
                        axspans_it.append(axspan)

                        this_result = {}
                        this_result['thresh'] = thresh
                        this_result['snr'] = snr
                        for d in descriptors:
                            this_result[d] = {
                                'default': getattr(ops, d)(im_, fgm_, **kwargs), # with mask without BG
                                'noFGM': getattr(ops, d)(im_, np.ones(im_.shape), **kwargs), # without mask without BG
                                'withBG': getattr(ops, d)(im_bg, fgm_, **kwargs), # with mask, without BG
                                'noFGMwithBG': getattr(ops, d)(im_bg, np.ones(im_.shape), **kwargs),
                                }
                        this_result['qc'] = self.qc_result(this_result)
                        results[str(it)][str(ic)] = this_result
                axspans.append(axspans_it)

        if self.verbose:
            raise NotImplementedError("4D featrue visualization is not implemented!")
            dfres = {   #'id': np.array([int(x) + 1 for x in results[str(it)].keys()]),
                        # bookkeep index -1 shift from earlier -.-
                        #'id': np.array([int(x) for x in results.keys()]),
                        'id': [np.array(list(v.keys()), dtype = np.int) for k, v in results.items()],
                        'qc': self.parse_result(        descriptor = 'qc',
                                                        value = None, result = results),
                        'thresh': self.parse_result(    descriptor = 'thresh',
                                                        value = None, result = results),
                        'snr': self.parse_result(        descriptor = 'snr',
                                                        value = None, result = results),
                        'Rg (-Bg)': self.parse_result(  descriptor = 'radius_of_gyration',
                                                        value = 'default', result = results),
                        'Rg (-Fgm-Bg)': self.parse_result(  descriptor = 'radius_of_gyration',
                                                        value = 'noFGM', result = results),
                        'Rg (+Bg)': self.parse_result(  descriptor = 'radius_of_gyration',
                                                        value = 'withBG', result = results),
                        'SI (-Bg)': self.parse_result(  descriptor = 'sum_intensity',
                                                        value = 'default', result = results),
                        'SI (-Fgm-Bg)': self.parse_result(  descriptor = 'sum_intensity',
                                                        value = 'noFGM', result = results),
                        'SI (+Bg)': self.parse_result(  descriptor = 'sum_intensity',
                                                        value = 'withBG', result = results)
                        } # dict for storing info for viz
            #for key in dfres: dfres[key] = np.insert(dfres[key], 0, 0) # add row for background

            #MH240723: Would have figure out how I can pass in features in 4D
            ## it is possible you can make a dict:
            ## {label: {feature: [0...N], ...}
            # transpose dictionary - thught this would work but it did not
            dfresT = {k: {} for k in dfres.keys()}
            for k in dfresT.keys():
                try:
                    dfresT[k] = dfres[k].T
                except AttributeError as e:
                    dfresT[k] = np.asarray(dfres[k]).T

            # pull out readout per unqiue ID
            ids = [np.unique(v)[0] for v in dfresT['id']]
            dfresF = {str(x):{} for x in sorted(ids)}
            for i, idx in enumerate(ids):
                dfresX = {k : v[i, :] for k,v in dfresT.items()}
                dfresF[str(idx)] = dfresX
            dfres = pd.DataFrame.from_dict(dfresF).T
            #dfres = dfres.drop('id', axis = 1)
            dfres = dfres.set_index('id', drop = False, inplace = False)
            #produces: ValueError: Length of values (1000) does not match length of index (1)
            mask = ops.map_labels(self.segmentor.mask)
            _ = masking.inspect_mask(   self.segmentor.stack.im, mask,
                                        self.segmentor.stack.pix2um,
                                        {'features': dfres})
        # TODO: cam play around with formatting but that is just comsetics
        if do_assign:
            self.results = results
            self.save()

    def qc_result(self, res):
        """ Quality control of results"""

        qc = 1 # 1 is passed
        # mark low snr
        if res['snr'] < 10: qc = 0
        # mark low size
        if res['radius_of_gyration']['default'] < 1.2: qc = 0

        return qc

    def parse_result(self, descriptor = 'radius_of_gyration',
                        value = 'default', itime = 0, icrop = None,
                        result = {}):
        """ Parse result of quantifier
        """
        doforce = False

        if not result:
            try:
                res = self.results
            except AttributeError as e:
                raise
        else:
            res = result

        if not itime:
            itime = list(res.keys())
        if icrop == None:
            icrop = [list(r.keys()) for (_, r) in res.items()]
        elif isinstance(icrop, (int, )):
            icrop = [[str(icrop)] for _ in range(len(itime))]
            doforce = True

        out = []
        for it in itime: # this may be a bit slow?
            icout = []
            for ic in icrop[int(it)]:
                # Option A:
                #if res[it][ic]['qc']: # if passed quality control
                #    out.append(res[it][ic][descriptor][value])
                #else:
                #    out.append(np.nan)

                # Option B - allows output of QC for later handling:
                if (int(ic) == 0) and not doforce: #MH240723 - forcing bg information
                    icout.append(None)
                    continue
                if value:
                    icout.append(res[it][ic][descriptor][value])
                else:
                    icout.append(res[it][ic][descriptor])
            out.append(icout)
        return np.asarray(out)

    def save(self, fpath = ""):
        if not fpath:
            fpath = self.segmentor.stack.fpath

        fpath = pth.splitext(fpath)[0]
        fpath = fpath + "_" + type(self).__name__
        dump_json(self.results, fpath)

    def load(self, fpath = ""):
        if not fpath:
            fpath = self.segmentor.stack.fpath
            fpath = pth.splitext(fpath)[0]
            fpath = glob(fpath + "*" + type(self).__name__ + "*.json")[0]
        self.results = load_json(fpath)
