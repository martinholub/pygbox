import numpy as np
import matplotlib.pyplot as plt
from pygbox import ops
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from os import path
import napari

def _show():
    plt.show()

def imshow(ax, x, gamma = 1, pixel_size = None, sat = 0, kwargs = {}):
    """ Plot an image

    Visualize an image in standardized way.
    Optionally applies gamma and contrast corrections and adds scale bar.

    Args:
        gamma (float): gamma correction exponent, x^gamma
        pixel_size (float): pixel size for scalebar (default: None)
        sat (float): fraction of pixels to saturate, must be smaller than 1 (default: 0)
        kwargs (dict): other params to pass to plt.imshow

    Returns:
        ax (plt Axes)

    """

    x = np.squeeze(x)
    nframes = int(np.prod(x.shape[2:]))
    x = np.reshape(x, x.shape[:2] + (nframes, ))

    x = ops.normalize(x) # normalize between 0 and 1 without stretching

    if gamma != 1:
        x = x ** gamma

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    kwargs_ = {'cmap': 'viridis'}
    kwargs_.update(kwargs)

    # saturate some pixels for visualization
    sat = sat if sat <=1 else sat/100
    vmax = 1 - sat
    # the extent will be [0, npixY, 0, npixX]
    if nframes == 1:
        ax.imshow(  x, vmin = 0, vmax = vmax, **kwargs_)
    else:
    #except TypeError as e:
        for i in range(nframes):
            ax.clear()
            ax.imshow(x[:,:,i], vmin = 0, vmax = vmax, **kwargs_)
            framenumber(ax, i)
            plt.draw()
            plt.waitforbuttonpress(0.2)

    #ax.axis('off')
    plt.tight_layout()

    if pixel_size:
        scalebar(ax, pixel_size)
    # plt.show()
    return ax

def framenumber(ax, i):
    """ Insert frame number on image in axes in standard way
    """
    txt = "frame: {}".format(int(i))
    fontprops = fm.FontProperties(size = 20, family = 'serif')
    ax.annotate(txt, xy = (0.95, 0.95), xycoords = 'axes fraction',
                fontproperties = fontprops, horizontalalignment = 'right',
                verticalalignment = 'top', color = 'white')

def scalebar(ax, pixel_size, length = 5, show_text = True):
    """ Insert scale bar on axis

    References
    ---------
    [1] https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar.html

    Author
    ---------------
    Martin Holub, scimartinh@gmail.com
    """

    if pixel_size:
        fontprops = fm.FontProperties(size = 20, family = 'serif')
        scb_col = 'white'
        scb_loc = 'lower right'
        scb_len = length / pixel_size # in data_units

        scb_txt = '{} $\mu$m'.format(length) if show_text else ''


        scb = AnchoredSizeBar(  ax.transData,
                                scb_len, scb_txt, scb_loc,
                                pad = 0.1,
                                color = scb_col,
                                frameon = False,
                                size_vertical = 4,
                                label_top = True,
                                fontproperties = fontprops)

        ax.add_artist(scb)


def spots(ax, x, y = [], kwargs = {}):
    """Overlay spots on image"""

    if len(x) == 0:
        return ax

    if len(y) == 0:
        y = x[:,1]
        x = x[:,0]

    kwargs_ = {
        'marker': 'o',
        'edgecolors': 'white',
        'facecolors': 'none',
        's': 100,
        'linewidth': 1,
        'alpha': 0.8
        }

    kwargs_.update(kwargs)
    ax.scatter(x, y, **kwargs_)
    plt.show()
    return ax

def fgm(ax, x):
    """Overlay foreground mask on axis"""
    if x.ndim > 2: x = np.squeeze(x)
    ax.contour( x, levels = 0,
                colors = 'red', linewidths = 2, linestyles = 'dashed')

    # invert boolean and mask out zeros
    x = ~x.astype(np.bool)
    x = np.ma.masked_where(x == 0, x)
    ax.imshow(x, alpha = 0.3, cmap = 'gray', clim = [0.5, 1])
    return ax

def savefig(ax, fname = 'test', suffix = '', kwargs = {}):
    """Save figure in standardized way"""

    fname, ext = path.splitext(fname)
    if ext == '.': ext = ''
    fname += suffix

    kwargs_ = {
        'format': 'svg' if not ext else ext[1:],
        'dpi': 600,
        'orientation': 'landscape',
        'transparent': True,
        'pad_inches': 0,
        }
    kwargs_.update(kwargs)

    if '.' not in fname:
        fname += '.' + kwargs_['format']

    if ax is None:
        ax = plt.gca()
    try:
        plt.sca(ax)
        fig = ax.get_figure()
    except Exception as e:
        fig = ax

    print(fname)
    plt.savefig(fname, **kwargs_)

def line_profile(x, coords):
    """Plot line profile through an image"""
    pass

def selectROI(ax):
    """tba"""
    pass


# 3D visualizer, with mask
from nibabel.viewers import OrthoSlicer3D
class MaskOrthoSlicer3D(OrthoSlicer3D):
    """
    Modified to display mask (if given) when `ctrl+M` pressed.

    References:
        [1]: https://nipy.org/nibabel/reference/nibabel.viewers.html#module-nibabel.viewers
        Author: Ramon, Martin
    """

    def __init__(self, data, mask=None, affine=None, axes=None, title=None):
        super().__init__(data, affine, axes, title)
        self._masktoggle = False
        self._mask = np.squeeze(mask)
        self._data = np.squeeze(data)
        del mask

    def _on_keypress(self, event):
        """Handle mpl keypress events"""
        if event.key is not None and 'escape' in event.key:
            self.close()
        elif event.key in ["=", '+']:
            # increment volume index
            new_idx = min(self._data_idx[3] + 1, self.n_volumes)
            self._set_volume_index(new_idx, update_slices=True)
            self._draw()
        elif event.key == '-':
            # decrement volume index
            new_idx = max(self._data_idx[3] - 1, 0)
            self._set_volume_index(new_idx, update_slices=True)
            self._draw()
        elif event.key == 'ctrl+x':
            self._cross = not self._cross
            self._draw()
        #This code is added:
        elif event.key == 'ctrl+m':
            if self._mask is not None:
                if self._masktoggle == False:
                    self._current_vol_data = self._mask
                    self._set_position(None, None, None, notify=False)
                    self.clim = [0,self._mask.max()]
                    self._draw()
                    self._masktoggle = True
                    print('Toggled to mask')
                elif self._masktoggle == True:
                    self._current_vol_data = self._data
                    self._set_position(None, None, None, notify=False)
                    self.clim = [0,self._data.max()]
                    self._draw()
                    self._masktoggle = False
                    print('Toggled to Image')

            else:
                print("mask == None, flapdrol")


def viz3d(im, mask=None, axaf = (1, 1, 1, 1)):
    """Overlay 3D image with mask"""

    #scale axes by real world sizes
    if not isinstance(axaf, (np.ndarray, )): axaf = np.asarray(axaf)
    if len(axaf) != 4:  axaf = np.append(axaf, [1] * (4-len(axaf)))
    axaf = np.identity(4) * (axaf)

    orthoslicer = MaskOrthoSlicer3D(im, mask, axaf)
    orthoslicer.cmap = 'viridis'
    orthoslicer.clim = [0, im.max()]
    orthoslicer.show()
    return orthoslicer

def mask_overlay(ax, im, mask, z = None, dx = None):
    """Convenience plotter combining `imshow` and `fgm` functions

    Args:
        z (int, float): plane number to plot (default: None)
            If < 1 then the plane number will be z*im.shape[2].
            If None then plots the central plane
        dx (array-like):
            Pixel size in 3 dimensions
    Returns:
        ax (plt Axes)
    """

    # Check dimensions
    if im.ndim > mask.ndim:
        mask = np.expand_dims(mask, -1) # expand last dimension
    assert im.shape == mask.shape, "Image and mask shapes must correspond!"

    # Select slice in time and space
    if im.ndim > 3:
        if im.shape[3] > 1:
            t = im.shape[3] //2
            im = im[:, :, :, t]
            mask = mask[:, :, :, t]

    if im.ndim > 2:
        if im.shape[2] > 1:
            if not z:
                z = im.shape[2] // 2
            elif z < 1:
                z = int(np.round(im.shape[2] * z) - 1)
            mask = mask[:, :, z]
            im = im[:, :, z]

    if mask.sum() == 0 or im.sum() == 0:
        print('Empty Image'); return

    # dropp all_zero rows and columns
    badcol = np.all(im == 0, 1).flatten()
    im = im[np.where(~badcol)[0], :]
    badrow = np.all(im == 0, 0).flatten()
    im = im[:, np.where(~badrow)[0]]

    mask = mask[np.where(~badcol)[0], :]
    mask = mask[:, np.where(~badrow)[0]]

    if mask.sum() == 0 or im.sum() == 0:
        print('Empty Image'); return

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # Visualize overlay
    ax = imshow(ax, im, gamma = 0.5, sat = 0.3, pixel_size = dx)
    ax = fgm(ax, mask)
    return ax
