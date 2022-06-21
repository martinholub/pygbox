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

    im = np.ma.masked_where(np.isnan(im), im)

    if how == 'max':
        return max_project(im, ax)
    elif how == 'sum':
        return sumi(im, None)
    elif how == 'meanmax':
        return meanmax_project(im, ax, **kwargs)
    else:
        raise NotImplementedError

def max_project(im, ax):
    """ Simple maximum intensity projection
    """
    return np.max(im, ax, keepdims = False)


def sum_intensity(im, mask = None, dx = None, do_bg_sub = False):
    """ Compute sum intensity
    """
    if mask is None: # sum in whole image
        mask = np.ones(im.shape, dtype = bool)

    mask = np.reshape(mask, im.shape) # correct for singleton dimensions

    if (im[mask==0].size > 0) & (do_bg_sub):
        im = bg_sub(im, mask)

    try:
        values = im[mask > 0]
    except Exception as e:
        print('Invalid mask. Returning NaN')
        return np.nan
    sumi = np.sum(values)
    return int(sumi)


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

def bg_sub(im, mask):
    """ Subtract background from an image knowing foreground mask
    """
    if not im.shape == mask.shape:
        mask = np.reshape(mask, im.shape)
    bg = im[np.logical_and(mask==0, im > .1)]
    bg_val = np.nanmedian(bg)
    bg_std = np.nanstd(bg)
    print("BG median = {:.1f}; BG std = {:.3f}".format(bg_val, bg_std))
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
