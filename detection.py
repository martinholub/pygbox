from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from scipy.ndimage import median_filter, maximum_filter, measurements, label
import numpy as np
import napari

try:
    from pygbox import ops
    from pygbox.masking import threshold_model
except ImportError as e:
    import ops
    from masking import threshold_model

def detect2D(   im, radius, rel_intensity_threshold = 1,
                rel_min_distance = 2.5, projection = 'max'):
    """ Detect objects in based on 2D information """

    # limit detection to a projected stack
    if im.ndim > 2:
        im_ = ops.project(im, projection, ax = -1)
    else:
        im_ = im
    im_ = np.ma.masked_where(im_ == 0, im_)
    # median filter to remove speckles
    im_ = median_filter(im_, size = 3)

    # find edges in gaussian smoothed image
    # NOTUSED: peak detection performs robustly on medfilt image
    # im_edge = gaussian_laplace(im_, sigma = radius//2)

    # minimal distance
    mindist = int(rel_min_distance * radius)
    # detect local minima
    spots = peak_local_max(im_, min_distance = mindist,
                exclude_border = (int(radius/2), ) * im_.ndim)
    xspot, yspot = (spots[:, 0], spots[:, 1])
    ## TODO?: blob finding at different scales

    ## exclude points that are too close from previously slected ones
    # not needed, cuz not doing substacks

    ## Average points that are closer than 1 diameter from each other
    # not needed, handeled by peak_local_max param min_distance

    ## sort out low intensity spots (could handle this directly in peak_local_max)
    try:
        keep = im_[xspot, yspot] > \
            (rel_intensity_threshold * np.nanmean(im_[im_>0]) \
             + 2*np.nanstd(im_[im>0]))
        xspot = xspot[keep]; yspot = yspot[keep]
    except IndexError as e:
        import pdb; pdb.set_trace()
        xspot = []; yspot = []; zspot = []

    ## discard spots that are too close to an edge
    # handeled in peak_local_max by exclude_border param

    ### DEPREC
    if im.ndim > 2:
        raise NotImplementedError("Passing 3D image to detect2D is not defined.")
        # recover z-depth by looking for maximum along z
        zspot = ops.get_z_pos(im[:,:,:,t], (xspot, yspot), radius)
        # Discard spots at the z-extremes:  (may not be needed)
        tf = (zspot > (nZ - radius/2)) | (zspot < radius/2)
        zspot = zspot[~tf]; xspot = xspot[~tf]; yspot = yspot[~tf]
    else:
        zspot = [0]*len(xspot)

    return (xspot, yspot, zspot)

def detect3D(im, radius, pix2um, opt = 1,
                rel_min_distance = 1, rel_intensity_threshold = 2):
    """ Detect objects in 3D data

    References:
        https://stackoverflow.com/a/55456970
    """
    im = np.ma.masked_where(im == 0, im)
    # I'll be looking in a 5x5x5 area
    #imf = maximum_filter(im, size= (radius, ) * im.ndim)
    imf = im

    # Threshold the image to find locations of interest
    _, thresh, _ = threshold_model(imf)

    if opt == 1:

        imbw = imf > thresh
        radius_ = radius # heurisitc
        imbw = remove_small_objects(imbw,
                    min_size = (radius_ ** 2) * (radius_ * pix2um[0]/pix2um[2]))

        # Since we're looking for maxima find areas greater than img_thresh
        labels, num_labels = label(imbw)

        # Get the positions of the maxima
        coords = measurements.center_of_mass(im, labels=labels, index=np.arange(1, num_labels + 1))

        # Get the maximum value in the labels
        values = measurements.maximum(im, labels=labels, index=np.arange(1, num_labels + 1))

        #could do some filtering based on brightness

        # return coorinates
        coords = np.asarray(coords)
    elif opt == 2:
        # minimal distance
        mindist = int(rel_min_distance * radius)
        # detect local minima
        imf = np.squeeze(imf)
        border_width = tuple(int(x) for x in np.around(radius * .75 / pix2um * pix2um[0]))
        coords = peak_local_max(imf, min_distance = mindist,
                    exclude_border = border_width,
                    threshold_abs = rel_intensity_threshold*thresh)
    else:
        raise NotImplementedError

    # manually confirm spots
    xspot, yspot, zspot = (coords[:, 0], coords[:, 1], coords[:, 2])
    return (xspot, yspot, zspot)

def rollnflip(x, ax = None):
    x = np.roll(x, 2, ax)
    x[:2] = x[:2][::-1]
    return x

def inspect_spots(stack, coords, radius = 20, spacing = (1., 1., 1.)):
    """ Edit spots interactively

    Notes:
        - on '~\Downloads\data\dev\pyth\*before*.tif' produces perfect detection
    """

    #stack = np.squeeze(np.moveaxis(stack, 2, 0))
    stack = np.moveaxis(stack, (2, 3), (1, 0))

    #coords = np.asarray([np.roll(np.round(x[:3]).astype(int), 1) for x in coords])
    coords = np.vstack(coords)
    coords = [rollnflip(x) for x in coords]
    #coords = np.array(coords)
    #corods = np.ma.masked_where(np.isnan(coords), coords)

    #spacing = np.roll(np.asarray(spacing), 1)
    # add time dimension to spacing, keeping the value small
    if len(spacing) < 4: spacing = np.append(spacing, (.01, )*(4 - len(spacing)))
    spacing = np.roll(spacing, 2); spacing[:2] = spacing[:2][::-1]

    viewer = napari.Viewer( title = "Spot Overlay",
                            axis_labels = ['t', 'z', 'x', 'y'], show = True)
    image = viewer.add_image(stack, scale = spacing, gamma = 0.5)
    points_layer = viewer.add_points(coords, size = radius,
                    face_color = 'magenta', edge_color = 'magenta',
                    scale = spacing, opacity = 0.5, n_dimensional = None)
    points_layer.out_of_slice_display = False
    if stack.shape[1] > 1: viewer.dims.ndisplay = 3

    napari.run() # anything after this will run once the window closes
    # get points, if you made any changes this will be reflected
    coords_ = points_layer.data.astype(int)
    coords_ = [np.concatenate((x[2:], x[:2][::-1])) for x in coords_]
    return coords_
