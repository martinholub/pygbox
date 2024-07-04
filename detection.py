from pygbox import ops
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from scipy.ndimage import median_filter, maximum_filter, measurements, label
import numpy as np
from pygbox.masking import threshold_model
import napari

def detect2D(   im, radius, rel_intensity_threshold = 1,
                rel_min_distance = 2.5, projection = 'max'):
    """ Detect objects in based on 2D information """

    # limit detection to a projected stack
    if im.ndim > 2:
        im_ = ops.project(im, projection, ax = -1)
    else:
        im_ = im
    # median filter to remove speckles
    im_ = median_filter(im_, size = 3)

    # find edges in gaussian smoothed image
    # NOTUSED: peak detection performs robustly on medfilt image
    # im_edge = gaussian_laplace(im_, sigma = radius//2)

    # minimal distance
    mindist = int(rel_min_distance * radius)
    # detect local minima
    spots = peak_local_max(im_, min_distance = mindist,
                exclude_border = (radius, ) * im_.ndim,
                indices = True)
    xspot, yspot = (spots[:, 0], spots[:, 1])

    ## TODO?: blob finding at different scales

    ## exclude points that are too close from previously slected ones
    # not needed, cuz not doing substacks

    ## Average points that are closer than 1 diameter from each other
    # not needed, handeled by peak_local_max param min_distance

    ## sort out low intensity spots (could handle this directly in peak_local_max)
    try:
        keep = im_[xspot, yspot] > \
            (rel_intensity_threshold*np.nanmean(im_) + 2*np.nanstd(im_))
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
                    indices = True, threshold_abs = rel_intensity_threshold*thresh)
    else:
        raise NotImplementedError

    # manually confirm spots
    coords_ = inspect_spots(imf, coords, radius, pix2um)

    xspot, yspot, zspot = (coords_[:, 0], coords_[:, 1], coords_[:, 2])
    return (xspot, yspot, zspot)


def inspect_spots(stack, coords, radius = 20, spacing = (1., 1., 1.)):
    """ Edit spots interactively

    Notes:
        - on '~\Downloads\data\dev\pyth\*before*.tif' produces perfect detection
    """
    stack = np.squeeze(np.moveaxis(stack, 2, 0))
    coords = np.asarray([np.roll(np.round(x[:3]).astype(int), 1) for x in coords])
    spacing = np.roll(np.asarray(spacing), 1)

    viewer = napari.Viewer( title = "Spot Overlay",
                            axis_labels = ['z', 'x', 'y'], show = True)
    image = viewer.add_image(stack, scale = spacing, gamma = 0.5)
    points_layer = viewer.add_points(coords, size = radius,
                    face_color = 'magenta', edge_color = 'magenta',
                    scale = spacing, opacity = 0.5, n_dimensional = True)

    napari.run() # anything after this will run once the window closes
    # get points, if you made any changes this will be reflected
    coords_ = np.roll(np.round(points_layer.data).astype(int), 2, axis = 1)

    return coords_
