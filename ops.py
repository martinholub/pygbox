import numpy as np
from scipy.ndimage.measurements import center_of_mass
import uuid

def project(im, how = 'max', ax = -1, kwargs = {}):
    """ Various projections of the data
    """

    assert isinstance(im, np.ndarray), 'Only np.ndarray can be projected.'
    if ax > im.ndim-1:
        ax = -1
        print('Incorrect axis, projecting along the last axis')
    how = how.lower()

    if im.shape[ax] == 1:
        print('Cannot project along singleton dimension, returning squeezed input.')
        return np.squeeze(im, axis = ax)

    im = np.ma.masked_where(np.isnan(im), im)

    if how == 'max':
        return max_project(im, ax = ax)
    elif how == 'sum':
        return sum_intensity(im, ax = ax, **kwargs)
    elif how == 'meanmax':
        return meanmax_project(im, ax = ax, **kwargs)
    else:
        raise NotImplementedError

def max_project(im, ax):
    """ Simple maximum intensity projection
    """
    return np.max(im, ax, keepdims = False)


def sum_intensity(im, mask = None, dx = None, ax = None, do_bg_sub = False):
    """ Compute sum intensity
    """
    if mask is None: # sum in whole image
        mask = np.ones(im.shape, dtype = bool)

    mask = np.reshape(mask, im.shape) # correct for singleton dimensions

    if (im[mask==0].size > 0) & (do_bg_sub):
        im = bg_sub(im, mask)

    if ax is not None:
        return np.sum(im, axis = ax, dtype = np.int)
    else:
        try:
            return int(np.sum(im[mask > 0]))
        except Exception as e:
            print('Invalid mask. Returning NaN')
            return np.nan


def meanmax_project(im, ax = -1, n = 5):
    """ Mean of few top brightest slices
    """
    im = np.sort(im, ax)[::-1] # sort descnding along last ax
    z_low = 1; z_high = min(n, im.shape[-1])
    maxp = np.mean(im[:,:,z_low:z_high], ax)
    return maxp


def normalize(x, a = 0, b = 1):
    """ Normalize data to an arbitrary A-B span
    """
    x = np.ma.masked_where(np.isnan(x), x)
    x = x.astype(np.float)
    x = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
    return x

def bg_sub(im, mask, how = "out"):
    """ Subtract background from an image knowing foreground mask
    """
    if not im.shape == mask.shape:
        mask = np.reshape(mask, im.shape)
    if how.lower() == "out":
        bg = im[np.logical_and(mask==0, im > .1)]
        bg_val = np.nanmedian(bg)
        bg_std = np.nanstd(bg)
        print("BG median = {:.1f}; BG std = {:.3f}".format(bg_val, bg_std))
    elif how.lower() == "in":
        bg = im[np.logical_and(mask==1, im > 0)]
        bg_val = np.quantile(bg, .1); bg_std = 0
        print("BG (10% quantile) = {:.1f}; BG std = {:.3f}".format(bg_val, bg_std))
    im = im - (bg_val + bg_std*2)
    im[im<0] = np.nan
    return im


def get_z_pos(im, XY, radius):
    """ Finds depth index in 3D array based on crop
    """
    #assert len(X) == len(Y), "Spot X and Y coordinates must be of same length!"
    if radius % 2: radius += 1
    X, Y = XY
    z_pos = np.zeros(len(X), int)

    for i in range(len(X)):
        nX, nY, nZ = im.shape[:3]
        span = np.arange(-radius/2, radius/2 + 1, dtype = int)
        Xid = X[i] + span
        Xid = np.delete(Xid, (Xid < 0) | (Xid > nX))
        Yid = Y[i] + span
        Yid = np.delete(Yid, (Yid < 0) | (Yid > nY))
        plane_sums = np.sum(im[np.ix_(Xid, Yid)], (0, 1))
        z_pos[i] = np.argmax(plane_sums)
    return z_pos

def mask_out(x, lo, hi, usenan = False):
    """ Masks out data outside low and high limits

    Parameters
    -----------------------------------
    lo: float
        low limit for masking
    hi: float
        high limit for masking

    Returns
    ------------------------------------
    x: np.ma.MaskedArray
        masked data
    keep_id: array_like
        indices of kept datapoints

    author: m.holub@tudelft.nl
    """

    x = x.astype(np.float) # cannot assign nan to int
    if usenan:
        x[np.bitwise_or(x<lo, x>hi)] = np.nan
        keep_id = np.argwhere(~np.isnan(x))
    else:
        x = np.ma.masked_where(x < lo, x)
        x = np.ma.masked_where(x > hi, x)
        keep_id = np.argwhere(~x.mask)
    return x, keep_id

def autocorrnd(x, which = 'corr', lag = 20, ntop = 1000):
    """ Autocorrelation of N-d data along first dimension

    Parameters
    ---------------------------------
    which: str
        which correlation to compute ['dif', or 'corr', default 'dif']
    lag: int
        Size of the look-ahead window
    ntop: int
        n-brightest pixels to calculate correlation on

    Returns
    ------------------------------------
    corrs/diffs: array_like
        correlations or differences
    """

    if which.lower().startswith('dif'):
        # This is not a proper corelation
        deltas = []

        x0 = x[0, keepid].flatten()
        x0 = x0 / np.max(x0)
        # sort array in increasing order, and take last ntop
        top_pixs = np.argsort(x0)[-ntop:]

        for i in range(1, x.shape[0]):
            x_ = x[i, keepid].flatten()
            x_ = x_ / np.max(x_)

            delta = np.sum(np.abs(x0[top_pixs] - x_[top_pixs]))
            deltas.append(delta)
        return deltas

    elif which.lower().startswith('cor'):
        corrs = []
        x0 = x[0, keepid].flatten()
        # sort array in increasing order, and take last ntop
        top_pixs = np.argsort(x0)[-ntop:]

        for i in range(0, x.shape[0] - lag):
            x0 = x[i, keepid].flatten()
            x0 = x0 / np.max(x0)

            corrs_ = []
            for j in range(0, lag):
                x_ = x[i+j+1,keepid].flatten()
                x_ = x_ / np.max(x_)

                corr = np.corrcoef(x0[top_pixs], x_[top_pixs])
                corrs_.append(corr[0,1])
            corrs.append(corrs_)
        return corrs
    else:
        raise NotImplementedError


def autocorr1d(x):
    """ Autocorrelation of 1D sequence """
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]
    if np.abs(r) > 0.5:
        print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    else:
        print('Appears not to be autocorrelated')
    return r, lag


def radius_of_gyration(im, mask, dx, do_bg_sub = True):
    """ Computes radius of gyration given image, mask and pixelsize.

    Args:
        im (ndarray): 3D array containing image
        mask (ndarray): 3D array containing mask
        dx (list): list containing pixelsize i.e. [dx,dy,dz]
        do_bg_sub: Whether to subtract background [default True]

    Returns:
        float64: radius of gyration value

    References:
        Author: Martin, Ramon
    """
    if np.all(mask == 0):
        #print("Mask array only consits of zero's")
        return np.nan

    mask = np.reshape(mask, im.shape) # correct for singleton dimensions

    # subtract background (presumably not done earlier)
    if (im[mask==0].size > 0) & (do_bg_sub):
        im = bg_sub(im, mask)

    im = np.squeeze(im); mask = np.squeeze(mask) # remove singleton dimensions
    # mask out background
    im = im*mask
    im = np.nan_to_num(im)
    # 3D center of mass
    cm = center_of_mass(im)
    if len(cm) == 2: cm += (0, ); im = np.expand_dims(im, 2)

    # Anisotropic coordinates w.r.t center of mass
    xx, yy, zz = aniso_mesh(im.shape, dx, cm, 'ij')
    # Squared radial distance from center of mass
    r2 = xx**2 + yy**2 + zz**2
    # radius if gyration - image weighted radial distance
    rg = np.sqrt(np.sum(r2*im)/np.sum(im))
    return rg

def aniso_mesh(nX, dx = (1, 1, 1), cm = (0, 0, 0), indexing = 'ij'):
    """ Generate regular but aniso-spaced 3d coordinate grid

    Args:
        nX (array-like): number of coordinate points to generate along each axis
        dX (array-like): optional (default = [1, 1, 1])
            distance between pixels in x, y, z directions in real-wolrd coordinates
        cm (array-like) optional (default = [0, 0, 0])
            location of center of mass

    Returns:
        (xx, yy, zz) (tupple of np.ndarray): anisotropic meshgrid

    """
    xx, yy, zz = np.meshgrid(   (np.arange(nX[0]) - cm[0]) * dx[0],
                                (np.arange(nX[1]) - cm[1]) * dx[1],
                                (np.arange(nX[2]) - cm[2]) * dx[2], indexing = indexing)
    return (xx, yy, zz)

def aniso_points(nX, dx = (1, 1, 1), cm = (0, 0, 0)):
    """Generate regular, but aniso-spaced points"""
    xx = (np.arange(nX[0]) - cm[0]) * dx[0]
    yy = (np.arange(nX[1]) - cm[1]) * dx[1]
    zz = (np.arange(nX[2]) - cm[2]) * dx[2]

    return xx, yy, zz

def cart2polar(x, y, z):
    """ Convert cartesian to spherical coordinates
    rho: radius, theta: polar, phi:azimuthal
    """
    rho = np.sqrt(x**2 + y**2 + z**2)
    #theta = -np.arccos(z/rho) # works too
    theta = np.arctan(z/np.sqrt(x**2 + y**2)) - np.pi/2
    phi = np.arctan2(y, x) + np.pi
    # Q: Why adding the angle np.pi or np.pi/2
    return rho, phi, theta

def polar2cart(rho, phi, theta, com = (0, 0, 0)):
    """ Convert spherical to cartesian coordinates"""
    x = rho * np.sin(theta) * np.cos(phi) + com[0]
    y = rho * np.sin(theta) * np.sin(phi) + com[1]
    z = rho * np.cos(theta) + com[2]
    return x, y, z

def reject_sigmas(x, n = 3):
    """ Reject tails assuming normal distribution

    Args:
        n (int, float): multiple of STD setting rejection boundary for both tails
    Returns:
        x (array_like): filtered values
        tf (array_like of bool): indices kept from original data
    """
    mu = np.nanmean(x)
    sig = np.nanstd(x)
    tf = (x > (mu - n*sig)) & (x < (mu + n*sig))
    x = x[tf]
    return x, tf

def get_mode(y):
    """ Get the most common value (i.e. mode) of distribution
    """
    nn, bins = np.histogram(y, 'fd', density = True)
    l_top_edge_idx = np.argmax(nn)
    bin_width = (bins[l_top_edge_idx + 1] - bins[l_top_edge_idx])
    mode = bins[l_top_edge_idx] + (bin_width / 2)
    return mode, bin_width

def map_labels(labels, mapping = None):
    """
    Map unique but arbitrary labels to labels defined by map

    Arguments
    -------------
    labels: np.ndarray
        n-dimensional array with unique labels
    mapping: dict, {old_label1: new_label1, ...}
        Mapping between labels. If None (0,1,2..N) incremental labels are used

    Returns
    ---------------
    out: np.ndarray
        n-dimensional array with remapped labels

    Notes
    ----------------
    [1]: https://gist.github.com/legaultmarc/92c4065e157eb93d57a5
    """

    labels_arr = np.unique(labels)
    if not mapping:
        mapping_arr = np.arange(0, len(labels_arr), dtype = int)
        mapping = {k:v for k,v in zip(labels_arr, mapping_arr)}
    else:
        mapping_arr = np.array([v for _,v in mapping.items()])


    target_type = set([type(x) for x in mapping_arr])
    target_type = target_type.pop()
    out = labels.astype(target_type)

    for i, (lab, target) in enumerate(mapping.items()):
        if lab == target: continue
        if target in labels_arr[i:]:
            temp_key = hash(str(uuid.uuid4()))
            out[labels == lab] = temp_key
            out[out == temp_key] = target
        else:
            out[labels == lab] = target

    return out

############# TRAPPING OPS
def simple_resample(x, y, dx = 1.0):
    """Downsamples data to given dx

    works when data is measured in mostly equal way, otherwise will be very lossy

    TODO: implement interpolation
    """
    dx_in = np.unique(np.diff(x))

    if len(dx_in) > 1:
        #raise ValueError('Uneven data sampling in `simple_resample`')
        return lin_interp(x, y, dx)
        #return np.asarray([np.nan]), np.asarray([np.nan])

    # no need to resample
    if dx_in[0] == dx:
        return x, y

    #downsample
    if dx_in[0] < dx:
        keepidx = ((x - x[0]) % dx) == 0
        return (x[keepidx], y[keepidx])
    else:
        #raise NotImplementedError("Upsampling currently not implemented.")
        return lin_interp(x, y, dx)
        #return np.asarray([np.nan]), np.asarray([np.nan])

def lin_interp(x, y, dx):
    """ linear interpolation """
    xp = np.arange(x[0], x[-1]+dx, dx)
    yp = np.interp(xp, x, y)
    return xp, yp


def pad_data(xydata, padval = np.nan):
    """Pad data to constant length

    Appends `padval` to 1D arrays to make the same length

    """
    maxlen = np.max([len(x) for x, _ in xydata])
    padwidth = [maxlen - len(x) for x, _ in xydata]

    xydata_out = []
    for w, (x, y) in zip(padwidth, xydata):
        xx = np.append(x, [padval]*w)
        yy = np.append(y, [padval]*w)
        xydata_out.append((xx, yy))
    return xydata_out

def summarize_data(y):
    """compute descriptive statistics"""
    # y-error
    n = np.count_nonzero(~np.isnan(y), axis = 0)
    ystd = np.nanstd(y, axis = 0)
    yerr = ystd / np.sqrt(n)

    # y-stat
    y = np.nanmean(y, axis = 0)

    return y, (ystd, yerr)

def relative_y(y):
    """
    Make y relative to (t0, y0)
    """
    idx = np.min(np.flatnonzero(~np.isnan(y) & (y > 0)))
    y0 = np.nanmean(y[idx:idx+3])
    return y / y0

def subset_data(data, maxX = (1, 60), maxY = (0, 2.5e4)):
    """Throw away data

    maxX: x axis is trimmed beyond this value
    maxY0: maximum value of y0; arrays exceeding it are dropped

    TODO: make the y handling smarter?, maybe you dont have to drop it really, if you are now averaging.
    """
    # conditions for keeping timesieres
    keepx = lambda x, maxX: x[-1] >= maxX # require time to run at least to maxX
    keepy = lambda y, maxY: y[0] <= maxY # TODO, find better way to account for few initial bad points

    # conditions for trimming timeseries (could be superseded)
    trimx = lambda x, maxX: x <= maxX # drop data beyond maxX
    #TODO: (may not be necessary, can acccount for by plot limits)

    trimdata = [(i, (x[trimx(x, maxX[1])], y[trimx(x, maxX[1])])) \
                for i, (x, y) in data \
                if (keepy(y, maxY[1]) & keepx(x, maxX[0]))]

    return trimdata

def normalize_trace(x, how = 'last', y = None):
    # discard any nans
    # x_ = x[~np.isnan(x)]

    # discard nans from ends of sequence
    x_ = x.copy()
    if any(np.isnan(x[0:3])):
        x_ = x_[np.argmax(~np.isnan(x_)):]
    # discard ending nans
    if any(np.isnan(x[-3:])):
        x_ = x_[:np.argmin(~np.isnan(x_))]
    if how.lower() == 'last':
        crp = len(x_)-int(len(x_)/10)
        xn = np.nanmean(x_[crp:])
    elif how.lower() == 'first':
        xn = np.nanmean(x_[0:4])
    if y is not None:
        x = y # normalize data in y by data in X
    return x / xn

def moving_average(x, w):

    if len(x) < w:
        print('Array len less than window size, skipping averaging')
        return x
    if w == 1:
        print('Requested win size is 1, skipping averaging')
        return x
    padl = w - 1
    padf = padl//2
    padb = padl - padf
    res = np.convolve(x, np.ones(w), "valid") / w
    res = np.pad(res, (padf, padb), constant_values = np.nan)
    return res
