import numpy as np
import pandas as pd
from copy import deepcopy, copy
import matplotlib.pyplot as plt
# for some reason this is the prefered method
from numpy.polynomial import polynomial as poly
import napari

from skimage.filters import threshold_otsu, rank, threshold_yen, threshold_mean
from skimage.filters import unsharp_mask, try_all_threshold
from skimage.filters import median as median_filter
from skimage import measure
from skimage.morphology import ball, disk, square, diamond, ball
from skimage.util import img_as_ubyte
from skimage.draw import polygon2mask

from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import uniform_filter, gaussian_filter, generate_binary_structure
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, binary_dilation
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
from scipy.stats import trim_mean, multivariate_normal
from scipy.signal import savgol_filter

try:
    from pygbox import viz
    from pygbox.viz.cm import gnuplot as Gnuplot
    from pygbox.ops import radius_of_gyration, aniso_points, aniso_mesh, normalize, polar2cart, cart2polar
except ImportError as e:
    import viz
    from viz.cm import gnuplot as Gnuplot
    from ops import radius_of_gyration, aniso_points, aniso_mesh, normalize, polar2cart, cart2polar

def _preprocess(im, opt = 1, args = {}):
    """Preprocess image for masking and visualization

    Args:
        opt (int): type of preprocessing
            1 - smoothing, 2 - sharpening, 3 - fourier filter (smoothing),
            4 - gamma correction & gaussian multiplication
    """
    if isinstance(opt, (list, tuple, )):
        for o in opt:
            im = _preprocess(im, o, args)
    else:
        if opt == 1: # smooth
            # works reasonably with BG SUB
            im = uniform_filter(im, size = 5)
            # Subtract background
            im = im - (np.median(im[im>0]))
            im[im < 0] = 0
        elif opt == 2: # sharpening
            # works reasonably (bit better than 1) without BG sub
            im = unsharp_mask(im, radius = 5, amount = 2, preserve_range = True)
            im = np.round(im).astype(int)
        elif opt == 3:
            im = fourier_filter(im, **args)
        elif opt == 4:
            # gamma correction to bring up faint regions
            im = (im**0.5)
        elif opt == 5:
            # gauss smoothing to force circular shape
            gauss = get_nD_gauss(im.shape, [x // 2 for x in im.shape])[0]
            im = im * gauss
        elif opt == 6:
            # decrease pixel per pixel variation
            im = gaussian_filter(im, sigma = 1/4, truncate = 3)
        else:
            raise NotImplementedError
            # add sparse-decon as an option

    return im

def get_nD_gauss(shape, sigma, mu = None):
    """ Get gaussian as nD array

    Args:
        shape (array_like): shape of the gaussian to generate
        sigma (array_lika): sigma for each dimenson
            if an int, same will be used for all dimensions
    Returns:
        val (np.ndarray): values of gaussian distribution
        axs (np.ndarray): points at which `val` were generated
    """

    axs = (np.arange(-x//2 + 1, x//2 + 1) for x in shape)
    if not mu:
        mu = np.asarray([0 for x in shape])
    if not isinstance(sigma, (list, tuple, )):
        sigma = [sigma] * len(mu)
    sigma = np.asarray(sigma)

    axs = np.meshgrid(*axs)
    xyz = np.column_stack([x.flatten() for x in axs])

    cov = np.diag(sigma**2)
    val = multivariate_normal.pdf(xyz, mean=mu, cov=cov)
    val = val.reshape(shape)

    val = val / val.max()

    return val, axs


def fourier_filter(im, cutoff = 25):
    """Filter in frequency space - leads to image smoothing

    Args:
        cutoff (int): Parameter controling level of smoothing
    Returns:
        im_filts (np.ndarray): Filtered image
    """

    if im.shape[2] > 1:
        #raise NotImplementedError("3D filtering not available, please pass image plane-by-plane!")
        print("3D filtering not available, filtering plane-by-plane.")

    #im = im - np.median(im) # subtrack background - probably not needed
    if im.ndim == 2: im = im[:, np.newaxis] # expand to third dimension

    im_filts = np.zeros(im.shape)
    for i in range(im.shape[2]):
        im_ = im[:, :, i]
        # compute 2D fft amd shift zerp-frequency component to center of the spectrum
        cfft_map = np.fft.fftshift(np.fft.fftn(im_))

        # build a mask
        rim_width = 10 # dont get what this one does
        shapes = cfft_map.shape
        axs = tuple([np.arange(s) for s in shapes])
        grids = np.meshgrid(*axs)
        rr = np.sum(np.asarray([x - int(y//2) for x, y in zip(grids, shapes)])**2, axis = 0)**0.5 # radial distances

        cfft_mask = np.zeros(shapes)
        cutoff_ = int(np.round(shapes[0] / cutoff)) # really dont get this one
        cfft_mask[rr < cutoff_] = 1
        cfft_mask[rr >= cutoff_] = np.exp(-((rr[rr>=cutoff_] - cutoff_) / rim_width)**2)

        im_filt = np.real(np.fft.ifftn(np.fft.fftshift(cfft_map * cfft_mask)))
        im_filts[:, :, i] = im_filt

    return im_filts

def mercator_mask(im, pix2um = (1, 1, 1)):
    """Radial projection based masking.

    Mercator masking attempts to create foreground mask. Assuming the object is
    roughly spherical, this can give good performance.

    Args:
        pix2um (array-like): pixel to micrometer conversion

    Returns:
        mask_spokes (np.ndarray): boolean mask
    """

    nX, nY, nZ = im.shape
    im = _preprocess(im, opt = 1)

    # find central object
    fg, bg, thr = mask_central_object(im)
    if fg.sum() == 0: return fg
    im = np.ma.masked_where(bg, im) #ignore background
    # get initial estimate of Rg for radial sampling axis range
    rg_quick = radius_of_gyration(im, fg, pix2um)
    cm = center_of_mass(im) # in pixel coordinates

    ## Sample radially
    n_rad = 40 # at how many distances to sample radially
    rad_span = 8 # how far from CoM to sample, relative to rg guess
    #   NB: maybe better to consider full-size of the original cube instead.
    skindepth = .1 # relative thickness of radial shell to sample at each step
    n_ang = 9 # at how many angles to sample radially and azimuthally, per pi
    # 9 seems to be enough, increase to 36 does not produce much change.

    # radial distances to sample
    rgs = np.arange(0, rad_span*rg_quick, 2*skindepth*rg_quick)
    rgs = np.append(rgs, [rad_span*rg_quick])
    vols = 4/3 * np.pi * rgs**3
    vols = np.linspace(vols[1], vols[-1], num = len(vols), endpoint = True)
    rgs = (3/(4*np.pi) * vols) ** (1/3)

    # Image cartesian coordinates
    xyz_points = aniso_points((nX, nY, nZ), pix2um)
    # Interpolators on cartesian image coordinates
    interp_fun = RegularGridInterpolator(xyz_points, im,
                        method = 'linear', bounds_error = False)
    interp_mask_fun = RegularGridInterpolator(xyz_points, fg,
                        method = 'linear', bounds_error = False)

    # Samplers on spherical coordinates
    theta_ax = np.linspace(-np.pi, 0, n_ang+1, endpoint = True) # polar coord sweep
    phi_ax = np.linspace(0, 2*np.pi, n_ang+1, endpoint = True) # azimuthal coord sweep
    # arrays for storing results
    spokes = np.zeros((len(theta_ax), len(phi_ax), len(rgs)))
    mask_spokes = np.zeros_like(spokes)
    for i, r in enumerate(rgs):
        # set up radial coordinate for this r
        r_prev = rgs[i-1] if i > 0 else r/3
        r_next = r+2*skindepth*rg_quick if i==(len(rgs)-1) else rgs[i+1]
        vol_ax =  np.linspace(4/3 * np.pi * r_prev**3, 4/3 * np.pi * r_next**3, 5, endpoint = True)
        rr_ax = (3/(4*np.pi) * vol_ax) ** (1/3)
        rr_ax = np.sort(np.append(rr_ax, [r])) # make sure r is there
        rr_ax = np.expand_dims(rr_ax, -1)

        # TODO: code is broken here at the moment.
        # Althoug the approach seems to make sense generates all 0 mask

        # Convert sampling coordinates to cartesian (in um)
        xx_shell, yy_shell, zz_shell = polar2cart(rr_ax, phi_ax, theta_ax, cm*pix2um)

        # Interpolate data from original (cartesian) to new (radially defined) positions
        xyz_points_sample = (xx_shell.T, yy_shell.T, zz_shell.T)
        mercator_map = interp_fun(xyz_points_sample)
        mercator_mask = interp_mask_fun(xyz_points_sample)
        if mercator_map.ndim > 1:# averoge over shell thickness
            # note that median([0, 1])==0.5
            spokes[:,:, i] = np.median(mercator_map, -1)
            mask_spokes[:, :, i] = np.median(mercator_mask, -1)
        else: # take values directly
            spokes[:,:, i] = mercator_map
            mask_spokes[:, :, i] = mercator_mask

    #plot_spokes(spokes, (theta_ax, phi_ax, rgs)) # debug

    # Threshold per spoke
    win = int(np.round(len(rgs)/4))
    for i, th in enumerate(theta_ax):
        for j, fi in enumerate(phi_ax):
            spoke = spokes[i, j, :] # single spoke
            spoke[np.isnan(spoke)] = 0 # for the next step ->
            spoke = np.trim_zeros(spoke, 'b') # crop off zero tail
            idx0 = len(spoke) - 1
            # derivative
            diff = np.diff(spoke)
            # Check if increasing for too long
            incr = np.append(diff > 0, 0) # make sure same length as input
            try:
                idx = np.min(np.argwhere((pd.Series(incr).rolling(win).sum() >= 3).values))
            except ValueError as e:
                idx = None
            # check if flat for too long
            try:
                idx2 = np.min(np.argwhere((pd.Series(diff).rolling(win).mean() > -35).values))
            except Exception as e:
                idx2 = None
            try:
                idx3 =  np.max(np.argwhere(spoke > thr))
            except Exception as e:
                idx3 = None
            # assign mask
            tf = np.ones(len(rgs), dtype = 'bool');
            tf[idx0:] = False
            #if idx: tf[idx:] = False
            #if idx2: tf[idx2:] = False
            if idx3: tf[idx3:] = False
            # TODO: still does not capture all what could happen; e.g. check object 0,1
            # otoh, works okish for smaller objects now, check object 5, 6
            # TODO: it actually does not seem to be doing very good masking, sadly :(
            # I think the problem is impossible to solve. Either you do in plane masking
            # that allows intontiguous objects in plane
            # which may kind of work, but will suffer from occasionally capturing an object that does not belong
            # or you could consider even then trying to fuse the incontigious masks.
            # this radial masking kind of works, but also cannot captrue thin processes
            # I can also try to do some image enhancement beforhand - maybe deconvo? Not sure tbh.

            mask_spokes[i, j, :] = tf
    ## Build 3D mask in cartesian coordinates
    # Interpolator on spherically defined coordinates
    tpr_points = (theta_ax, phi_ax, rgs)
    interp_mask_fun = RegularGridInterpolator(tpr_points, mask_spokes,
                        method = 'linear', bounds_error = False)

    # irregular grid interpolation - veery slow - produces similar results to grid interp.
    #tpr_mesh = np.meshgrid(*tpr_points, indexing = 'ij')
    #tpr_points =  np.vstack(map(np.ravel, tpr_mesh)).T
    #mask_spokes = np.meshgrid(mask_spokes, indexing = 'ij')[0]
    #interp_mask_fun = RBFInterpolator(tpr_points, mask_spokes, neighbors = 4**3)

    # Map spokes back to fully-sized cartesian grid
    # convert cartesian to radial coordinates
    xyz_mesh = aniso_mesh((nX, nY, nZ), pix2um, cm) # it does matter wheter u use xy or ij idnexing here - idk yet which one is the correct one :()
    rrs, phis, thetas = cart2polar(*xyz_mesh)
    tpr_sample = np.vstack(map(np.ravel, (thetas, phis, rrs))).T

    # Carry out the interpolation
    try:
        mask_spokes = interp_mask_fun(tpr_sample)
    except Exception as e:
        mask_spokes = np.zeros(tpr_sample.shape[0], dtype = 'bool')
    mask_spokes[np.isnan(mask_spokes) == 1] = 0

    # threshold spokes to make sure they are binary
    mask_spokes[mask_spokes <= .5] = 0
    mask_spokes[mask_spokes > .5] = 1
    mask_spokes = mask_spokes.astype('bool')

    # reshape to 3D
    mask_spokes = np.reshape(mask_spokes, im.shape)
    # fil holes in the mask
    mask_spokes = binary_closing(mask_spokes, np.ones((4, ) * mask_spokes.ndim))

    # Later, consider passing as sparse, or list of x,y,z points
    # return measure.find_contours(binary_spokes)
    return mask_spokes


def planar_mask(im, pix2um = (1, 1, 1)):
    """Construct 3D mask by plane by plane masking

    Args:
        im (np.ndarray): 2- or 3-D array to mask

    Returns:
        fgms (np.ndarray): boolean mask

    """
    nX, nY, nZ = im.shape
    im_ = _preprocess(im, opt = [6])
    #im_ = im # MH 240704
    im_min_trhesh, thresh, snr = imthresh(im_) # get a threshold from 3D volume

    # this is natural but somewhat does not seem to work well
    char_size = int(((im_min_trhesh > 0).sum() * 3 / 4 / np.pi)**(1/3))
    # This actually works
    char_sum = im_min_trhesh[im_min_trhesh > 0].sum()
    char_size = int(char_sum / 3e3)

    #print('Char size is: {}'.format(char_size))
    #print('Sum is is: {}'.format(im_min_trhesh[im_min_trhesh > 0].sum()))
    # note that SNR and thresh have been obtained on preprocessed data!

    #im = np.ma.masked_where(np.isnan(im), im) #ignore nans
    fgms = np.zeros((nX, nY, nZ), dtype = 'bool')
    #import pdb; pdb.set_trace()
    for iz in range(nZ):
        imiz = im_min_trhesh[:, :, iz] # extract plane
        fgm = imiz > 0

        # remove tiny regions
        fgm = binary_opening(fgm, disk(3), iterations = 1) # MH 240704

        # connect fragmented regions
        #fgm = binary_closing(fgm, diamond(5)) # 15 .. 21 works fine

        # remove small regions
        #fgm = binary_opening(fgm, disk(5))

        # close dark holes
        #fgm = binary_closing(fgm, square(3))
        #fgm = binary_fill_holes(fgm, square(3))

        # pick central object - applying only later, to 3D reslt
        #fgm = mask_central_object(fgm)[0]

        #dilate
        #for _ in range(char_size // 2): # around 3 .. 5 rounds of dil work fine
        fgm = binary_dilation(fgm, disk(3), iterations = 4)
        #fgm = dilate_mask(imiz, fgm)

        # fill holes in mask
        fgm = binary_fill_holes(fgm)

        if fgm.sum() > 0: # smooth boundary
            fgm = smooth_boundary(fgm)

        fgms[:, :, iz] = fgm

    try:
        fgms = mask_central_object(fgms)[0]
    except ValueError as e: # does not happen
        print("Could not find central object in the mask!")
        import pdb; pdb.set_trace()

    #MH-18052022 - this helps, in Z, likely due to pixel size assymetry
    try:
        fgms = binary_dilation(fgms, ball(3), iterations = 1)
    except Exception as e:
        print("planar_masking.binary_dilation: 3D Dilating with diamond")
        fgms = binary_dilation(fgms, iterations = 1)

    #if nZ > 1: fgms = dilate_mask_z(fgms)

    # Verbose inspection
    # if fgms.sum() > 0:
    #     _ = inspect_mask([im_, im_min_trhesh, im], fgms,
    #     labels_kwargs = {'color': {1: 'white'}}, scale = pix2um)

    #if fgms.sum() > .75*fgms.size:
        # dont tust masks that cover more than BG
        #fgms = np.zeros_like(fgms)

    return fgms

def planar_mask_lambda(im, pix2um = (1, 1, 1)):
    """Construct 3D mask by plane by plane masking

    Args:
        im (np.ndarray): 2- or 3-D array to mask

    Returns:
        fgms (np.ndarray): boolean mask

    """
    nX, nY, nZ = im.shape
    im_ = _preprocess(im, opt = [ 6, 4 ])
    #im_ = im

    im_min_trhesh, thresh, snr = imthresh(im_) # get a threshold from 3D volume

    # this is natural but somewhat does not seem to work well
    char_size = int(((im_min_trhesh > 0).sum() * 3 / 4 / np.pi)**(1/3))
    # This actually works
    char_sum = im_min_trhesh[im_min_trhesh > 0].sum()
    char_size = int(char_sum / 3e3)

    #print('Char size is: {}'.format(char_size))
    #print('Sum is is: {}'.format(im_min_trhesh[im_min_trhesh > 0].sum()))
    # note that SNR and thresh have been obtained on preprocessed data!

    #im = np.ma.masked_where(np.isnan(im), im) #ignore nans
    fgms = np.zeros((nX, nY, nZ), dtype = 'bool')
    #import pdb; pdb.set_trace()
    for iz in range(nZ):
        imiz = im_min_trhesh[:, :, iz] # extract plane
        fgm = imiz > 0

        # remove tiny regions
        #fgm = binary_opening(fgm, disk(3), iterations = 2)
        fgm = binary_opening(fgm, iterations = 2)

        # connect fragmented regions
        #fgm = binary_closing(fgm, diamond(5)) # 15 .. 21 works fine

        # remove small regions
        #fgm = binary_opening(fgm, disk(5))

        # close dark holes
        #fgm = binary_closing(fgm, square(3))
        #fgm = binary_fill_holes(fgm, square(3))

        # pick central object - applying only later, to 3D reslt
        #fgm = mask_central_object(fgm)[0]

        #dilate
        #for _ in range(char_size // 2): # around 3 .. 5 rounds of dil work fine
        #fgm = binary_dilation(fgm, disk(3), iterations = 4)
        #fgm = dilate_mask(imiz, fgm)

        fgm = binary_dilation(fgm,iterations=2)

        # fill holes in mask
        fgm = binary_fill_holes(fgm)

        if fgm.sum() > 0: # smooth boundary
            fgm = smooth_boundary(fgm)

        fgms[:, :, iz] = fgm

    try:
        fgms = mask_central_object(fgms)[0]
    except ValueError as e: # does not happen
        print("Could not find central object in the mask!")
        import pdb; pdb.set_trace()

    #MH-18052022 - this helps, in Z, likely due to pixel size assymetry
    try:
        fgms = binary_dilation(fgms, ball(1), iterations = 1)
    except Exception as e:
        print("planar_masking.binary_dilation: 3D Dilating with diamond")
        fgms = binary_dilation(fgms, iterations = 1)

    #fgms = dilate_mask_z(fgms)

    # Verbose inspection
    # if fgms.sum() > 0:
    #     _ = inspect_mask([im_, im_min_trhesh, im], fgms,
    #     labels_kwargs = {'color': {1: 'white'}}, scale = pix2um)

    #if fgms.sum() > .75*fgms.size:
        # dont tust masks that cover more than BG
        #fgms = np.zeros_like(fgms)

    return fgms

def mask_central_object(im):
    """Keep only central object in 2D boolean mask

    Args:
        im (np.ndarray): boolean mask to use
    Returns:
        im_fg (bool np.ndarray): foreground mask
        im_bg (bool np.ndarray): background mask
        thr (float): threshold (if used to make boolean array)
    """

    centr = [int(x/2) for x in im.shape]
    if im.dtype != 'bool': # get boolean mask if not yet
        im_thr, thr, _ = imthresh(im)
        im_fg = im_thr > 0
    else:
        im_fg = im
        thr = None

    # remove tiny regions
    #im_fg = binary_opening(im_fg, np.ones((4, ) * im_fg.ndim)) # will be bool
    # connect fragmented regions
    #im_fg = binary_closing(im_fg, np.ones((9, ) * im_fg.ndim)) # will be bool

    # Select one central/biggest object if multiple found
    labels, n_labels = measure.label(im_fg, return_num = True)
    regprops = measure.regionprops(labels)

    if n_labels > 1: # 0,1 are there always if some bg found, but 0 does not count towards n_labels

        # index of biggest region
        #idx = np.argmax([l.area for l in regprops])

        #idx of region closest to center.
        centrs = [l.centroid for l in regprops]
        idx = np.argmin(np.sum((np.array(centr)-np.array(centrs)) ** 2, axis = 1))

        # create mask with a single label
        im_fg = np.zeros_like(im_fg, dtype = 'bool')
        im_fg[labels == regprops[idx].label] = True

    im_bg = ~im_fg
    return im_fg, im_bg, thr


def imthresh(im):
    """Find intensity threshold value and subtract it from an image

    TODO: compare the custom algo to off-the-shelf thresholding algos and pick
    the off-the-shelf that works the best

    Returns:
        thresh (float): threshold value
        im_out (np.ndarray): image with threshold subtracted

    References:
        [1] https://scikit-image.org/docs/dev/auto_examples/applications/plot_thresholding.html
    """

    # global - takes up bright part of nucleoid
    #thresh = threshold_otsu(im) # usually ~2x of mean thresh
    #thresh = threshold_mean(im) # works quite well
    _, thresh, snr = threshold_model(im)

    # try:
    #     fig, ax = try_all_threshold(im)
    # except TypeError as e:
    #     fig, ax = try_all_threshold(im[:,:, im.shape[2]//2])
    # plt.show()

    #local - takes up most of the image as foreground and is slow
    #im = normalize(im, 0, 255); im = im.astype(np.uint8)
    #thresh = rank.otsu(im, footprint = ball(radius = 5))

    # apply threshold
    im_out = im - thresh
    im_out[im_out <= 0] = 0

    return (im_out, thresh, snr)

def threshold_model(im):
    """ Generates threshold for an image based on background model

    Get treshold of an array, image or stack by scaled intensity sorting.
    Both sorting index axis and brightness axis are normalized. Then, from
    each apex, points along half of the axis length are used for a linear fit.
    these two fits yield a crossing point. The raw curve point closest to this point(in
    scaled xy coordinates) yields the treshold value.  A tentative SR-ratio is
    calculated by the crossing point of the two fits with the index=max axis.

    Args:
        im (array_like): data to threshold

    Returns:
        im (arra_like): thresholded image
        thresh (float): threshold
        snr (float): signal to noise ratio

    References:
        Thesis Natalia Vtyurina (Written by Margreet Docter, re-edited JacobKers 2020)
    Authors:
        Martin Holub, Jacob Kerssemakers, Margreet Docter
    """

    # sort intensity value
    im = im.astype(float)
    sim = np.sort(im.flatten())
    sim = sim[~np.isnan(sim)]

    if len(np.unique(sim)) < 2: return (im, 0, np.nan)

    # make intensity-axis (Y) scale approx as index-axis (X)
    int_scaler = sim.size / sim.max()
    sim *= int_scaler

    # fit a line to lower half number of pixels
    n_half = int(sim.size / 2)
    x_lo = np.arange(n_half)
    fit_lo = poly.polyfit(x_lo, sim[:n_half], deg = 1)
    fit_lo_y = poly.polyval(x_lo, fit_lo) # evaluate for latter plotting

    # fit a line to pixels brighter than half of maximum
    n_bright = np.sum(sim > (sim.max() / 2))
    x_up = np.arange(sim.size - n_bright+1, sim.size+1)
    fit_up = poly.polyfit(x_up, sim[-(n_bright):], deg = 1)
    fit_up_y = poly.polyval(x_up, fit_up) # evaluate for later plotting

    # find crossing point of the two line
    xc = (fit_lo[0] - fit_up[0]) / (fit_up[1] - fit_lo[1])
    yc = poly.polyval(xc, fit_lo) # fit_lo[1]*xc + fit_lo[0]
    # find I(x) knee, closest to the crossing point [xc, yc]
    # becayse I is scaled in X - pos and int have equal weight
    dist = np.sqrt((np.arange(sim.size) - xc)**2 + (sim - yc)**2)
    # get threshold value and correct for scaling
    thresh_sim = sim[np.argmin(dist)]
    thresh =  thresh_sim / int_scaler
    # calculate SNR
    snr = poly.polyval(sim.size - 1, fit_up) / poly.polyval(sim.size - 1, fit_lo)

    # apply threshold
    im_out = im - thresh
    im_out[im_out <= 0] = 0

    sim_to_plot = sim[np.logical_and(sim < 5*thresh_sim, sim > 0)]
    #import pdb; pdb.set_trace()
    #_plot_threshold_model_fit(sim, (fit_lo, fit_up), [(np.argmin(dist), sim[np.argmin(dist)])], thresh_sim)
    #_plot_threshold_model_fit(sim, ((x_lo, fit_lo_y), (x_up, fit_up_y)), [(np.argmin(dist), sim[np.argmin(dist)])])

    return (im_out, thresh, snr)

def _plot_threshold_model_fit(x, fits, pts, thresh = None):
    """ Helper function for plotting of thresholded fit
    """
    # standardized figure
    viz.plots.set_plot_style()
    fig, ax = viz.plots.make_fig()

    legs = ['data', 'fit faint', 'fit bright', 'threshold', 'fit crossing']
    # Generate plot
    xx = np.arange(x.size)
    ax.plot(xx, x, color = 'gray')
    for f, fmt in zip(fits, ('k-.', 'k--')):
        if (f[0]).size == 1:
            ax.plot(xx, poly.polyval(xx, f), fmt)
        else:
            ax.plot(f[0], f[1], fmt)


    #ptsx = [x[0] for x in pts]
    #ptsy = [y[1] for y in pts]
    #ax.plot(ptsx, ptsy, 'ro')

    for pt, fmt in zip(pts, ('ro', 'k*')):
        ax.plot(pt[0], pt[1], fmt)

    ax.legend(legs[:(len(fits) + len(pts) + 1)])
    if not thresh:
        ymax = 1.025 * x.max()
    else:
        ymax = 5*thresh
    ax.set_ylim(bottom = 0, top = ymax)
    ax.set_xlim(left = 0)

    # hide tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # hide ticks
    plt.xticks([])
    plt.yticks([])

    ax.set_xlabel('sorting index')
    ax.set_ylabel('sorted intensity values [a.u.]')
    # log scale
    #ax.set_xscale('log')

    # show or save
    #plt.show()
    viz.viz.savefig(ax, 'threshold_model.svg')


def dilate_mask(im, mask, step_size = 1):
    """Iteratively dilate mask to capture most of signal

    Args:
        step_size (int): how many consecutive dilations before checking for condition
    """

    # What do you use as measure of signal?
    bg = im[~mask]; bgmean = trim_mean(bg, .01)
    fg = im[mask]; fgmean = trim_mean(fg, .01)
    fg1mean = fgmean
    if np.isnan(fg1mean): return mask
    if bgmean <= .001: return mask
    ctr = 0
    while fg1mean > (1.5*bgmean):
        # before update
        idx0 = np.flatnonzero(mask)

        # update
        for _ in range(step_size):
            mask = binary_dilation(mask, disk(radius=3))

        idx1 = np.flatnonzero(mask)
        idx = idx1.ravel()[np.isin(idx1, idx0, invert = True)]

        fg1 = im.ravel()[idx]; fg1mean = trim_mean(fg, .01)

        #mask_diff = np.bitwise_xor(mask0, mask)
        ctr += 1
        if ctr > 20:
            break
    return mask

def dilate_mask_z(mask, opt = 1):
    """Unidirectional mask dilation in z-direction
    """
    mshape = mask.shape
    mask = np.squeeze(mask)
    # Option 1
    if opt == 1:
        #struct = np.vstack([[0]*3, [1]*3, [0]*3]).T
        #struct = generate_binary_structure(3, 1)
        struct = np.zeros((3,3,3))
        struct[1, 1, :] = 1
        mask = binary_dilation(mask, struct)
    elif opt == 2:
        # option 2

        z_sums = np.sum(mask, ax = [1, 2])
        bottom, top = np.argwhere(z_sums > 0)[[0, -1]]
        mask[..., bottom - 1] = mask[..., bottom]
        mask[..., top + 1] = mask[..., top]
    else:
        raise NotImplementedError

    mask = np.reshape(mask, mshape)
    return mask



def smooth_boundary(mask):
    """Smooth mask boundary by applying Savitzky Golay Filter

    You will have to close all holes before applying the filter.

    Returns:
        mask (np.ndarray): smoothed mask

    References:
        # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        # https://bartwronski.com/2021/11/03/study-of-smoothing-filters-savitzky-golay-filters/
    """

    try:
        bnd = measure.find_contours(mask)[0]
    except IndexError as e: # not sure why this happens if fgm.sum()>0
        return mask
    # boundary smoothing produces corrupt mask if mask on the edge already
    # maybe can fix this with using different 'mode'
    touches_edge = (np.logical_or(bnd.flatten() == 0, bnd.flatten() >= (mask.shape[0]-1))).sum()
    if touches_edge: return mask

    bnd_length = len(bnd)
    # MH: parameters here are somewhat arbitrary - works quite well with range of params
    win_length = bnd_length // 4
    if not np.mod(win_length, 2): win_length += 1 # most be odd
    savgol_params = {'polyorder': 8, 'mode': 'interp'}

    if win_length > 7:
        #x_out = savgol_filter(np.repeat(bnd[:, 0], 2), win_length, **savgol_params)
        #y_out = savgol_filter(np.repeat(bnd[:, 1], 2), win_length, **savgol_params)
        #bnd_smooth = np.asarray([x_out[:bnd_length], y_out[:bnd_length]]).T

        x_out = savgol_filter(bnd[:, 0], win_length, **savgol_params)
        y_out = savgol_filter(bnd[:, 1], win_length, **savgol_params)
        bnd_smooth = np.asarray([x_out, y_out]).T
        # convert list of points to a closed boolean shape
        mask = polygon2mask(mask.shape, bnd_smooth)
    return mask

def qc_mask3D(mask):
    """ Quality control of mask """

    is_ok = True
    # Aspect Ratio
    ## For mow simple - could also calculate max distance of points on contour as measure of size

    ## Find largest horizontal plane, get its size. Approximate it by circle.
    xysum = np.sum(mask, axis = (0, 1)); maxxysum = np.max(xysum)
    rxy = np.sqrt(maxxysum) / np.pi

    ## Find largest vertical plane, get its size. Approximate it by circle.
    xzsum = np.sum(mask, axis = (0, 2)); maxxzsum = np.max(xzsum)
    yzsum = np.sum(mask, axis = (1, 2)); maxyzsum = np.max(yzsum)
    rxz = np.sqrt(np.max([maxxzsum, maxyzsum])) / np.pi

    ## Calculate metric
    try:
        ar = rxy / rxz
        if (ar > 2) or (ar < 1/2) or (np.isnan(ar)): is_ok = False
    except Exception as e:
        is_ok = False

    # Mask Size
    ## For now simple - using just 2D info
    if rxy < 8: is_ok = False # 8 is bit of a guess but works well
    #print('Rxy is {:.3f}'.format(rxy))

    return is_ok

def inspect_mask(stack, mask, spacing = (1., 1., 1.),
                    labels_kwargs = {}, spans = None):
    """ Edit mask interactively

    Notes:
        -Pass Z-axis at 3rd position, it will be rolled to the beginning

    """
    if not isinstance(stack, (tuple, list)): stack = [stack]
    # confirm/edit spots with napari
    mask_shape = mask.shape
    #stack = [np.squeeze(np.moveaxis(s, (2, 3), (1, 0))) for s in stack]
    #mask = np.squeeze(np.moveaxis(mask, (2, 3), (1, 0)))
    stack = [np.moveaxis(s, (2, 3), (1, 0)) for s in stack]
    mask = np.moveaxis(mask, (2, 3), (1, 0))

    # add time dimension to spacing, keeping the value small
    if len(spacing) < 4: spacing = np.append(spacing, (.1, )*(4 - len(spacing)))
    spacing = np.roll(spacing, 2); spacing[:2] = spacing[:2][::-1]

    viewer = napari.Viewer( title = "Mask Overlay",
                            axis_labels = ['t', 'z', 'x', 'y'], show = True)

    cmaps = ['viridis', Gnuplot, 'magma']
    for stk, cmap in zip(stack, cmaps):
        image_layer = viewer.add_image(stk, scale = spacing,
                            gamma = 0.5, colormap = cmap)
    labels_layer = viewer.add_labels(mask, scale = spacing, opacity = 0.5,
                    **labels_kwargs)
    labels_layer.n_edit_dimensions = 3
    labels_layer.preserve_labels = False
    labels_layer.brush_size = 10

    if spans is not None:
        bbox_pts = [make_bbox_array(span) for span in spans]
        # option 1 - napari does not support cuboids yet
        # shapes_layer = viewer.add_shapes(
        #             bbox_pts, face_color='transparent',
        #             edge_color='green', name='bbox')

        # option 2 - render vertices as points - difficult to interpret
        bbox_pts = np.vstack(bbox_pts)
        points_layer = viewer.add_points(bbox_pts, size = 10,
                        face_color = 'green', edge_color = 'green',
                        scale = spacing, opacity = 1, n_dimensional = True,
                        symbol = 'cross', name='bbox-vertices')

        # option 3 render as a new label layer

    napari.run() # anything after this will run once the window closes
    # may have to run this from command line!

    # get (updated) mask
    mask = labels_layer.data
    if mask.ndim == 2: mask = np.reshape(mask, (1, ) + mask.shape)
    mask = np.moveaxis(mask, (1, 0), (2, 3))
    # TODO: find a way how to get mask from newly added layer
    return mask

def radial_mask(im, radius, pix2um):
    """ Assign pixels to mask radially """
    centr = np.array(im.shape[:3])//2 # is already centred

    # Anisotropic coordinates w.r.t coordinate center
    xx, yy, zz = aniso_mesh(im.shape, pix2um, centr, 'ij')
    # Squared radial distance
    r2 = xx**2 + yy**2 + zz**2
    #import pdb; pdb.set_trace()
    # asign mask within radius
    mask = np.zeros(im.shape[:3], dtype = 'bool')
    mask[r2 < (radius/3)**2] = True

    return mask

def recenter_corners(mask, exts = None, max_spans = None):
    """
    Recalculate mask corners
    """
    if max_spans is None:
        max_spans = mask.shape
    spans = []
    regprops = measure.regionprops(mask)
    for r in regprops:
        if exts is None:
            #raise NotImplementedError("Must supply extents")
            spans.append(get_bbox(r))
        centr = r.centroid
        spans.append(get_span_axs(exts, centr, max_spans))
    return spans

def get_bbox(r):
    """get bounding box from a regionproperties object """
    bbox = r.bbox
    bbox = np.asarray([(i,j) for i,j in zip(bbox[:3], bbox[3:])])
    return bbox

def make_bbox_array(crn):
    minx = crn[0][0]; maxx = crn[0][1]
    miny = crn[1][0]; maxy = crn[1][1]
    minz = crn[2][0]; maxz = crn[2][1]

    bbox_pts = np.array(
        [[minx, miny, minz], [maxx, miny, minz], [maxx, maxy, minz], [minx, maxy, minz],
        [ minx, miny, maxz], [maxx, miny, maxz], [maxx, maxy, maxz], [minx, maxy, maxz]]
    )

    #bbox_pts = np.moveaxis(bbox_pts, 2, 0)
    bbox_pts = np.roll(bbox_pts, 1, 1)
    return bbox_pts

def get_span_ax(ex, cc, nn):
    """
    Parameters
    --------------
        ex: int, extent to span in pixels
        cc: int, centroid coordinate
        nn: int, maximal value for span

    Returns
    -------------------
        span: list, span [from, to]
    """
    ax_low = np.max([cc - ex//2, 0])
    ax_high = np.min([nn, cc + ex//2])
    # Force centering around the object
    if ax_low == 0: ax_high = np.min([ax_high, 2*cc - ax_low])
    #if ax_high == nn: ax_low = np.max([ax_low, 2*cc - ax_high])

    if ax_high == 0: ax_high += 1
    return [int(np.around(ax_low)), int(np.around(ax_high))]

def get_span_axs(exts, centr, max_spans):
    """calculate ax span for all axes

    References
    ---------------
    get_span_ax
    """
    c_spans = []
    for ex, cc, nn in zip(exts, centr, max_spans):
        c_spans.append(get_span_ax(ex, cc, nn))
    return c_spans
