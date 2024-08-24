""" perimeter and intnsnity tools 
* aim: get a signature of how corrugated a pattern is.
* approach: use area of a disk with equal Rg 
* steps: 
    - use the radius-of-gyration for reference and a max-projection of a stack to work on (this will avoid overdetecting holes, for example).
    - We then binarize the pattern with a treshold such, that the binarized surface has the same area as that of a disk with the same radius of gyration (note: for a disk, unlike a ring, this R is sqrt(2) larger than Rg!.
    - count number of pixels on edge
    - scale this on the area of the equivalent disk
* interpretation: 0 is disk-like, 1 is a cloud of all isolated pixels 

context: traps data, thesis Martin Holub 2024
# Jacob Kers, 2024
"""

from skimage.morphology import  disk
from scipy.ndimage import binary_opening, binary_erosion
import cv2
import numpy as np
import math as mt

def get_dna_mask(ori_im, treshold):
        #get a binary image from an image and a treshold 
        okay_im=np.max(ori_im)>0 and np.std(ori_im)>0
        if okay_im:  
            dna_im=ori_im
            dna_im= dna_im-treshold
            dna_im[dna_im<0]=0
            binary_dna_im=dna_im>0
            binary_dna_im=binary_opening(binary_dna_im, disk(1), iterations = 1)
        else:
            binary_dna_im=0*ori_im
        return binary_dna_im


def get_treshold_data(data_in):
    """ classic triangulation tresholding, equal importance to intensity axis and sorting axis
    works best with a majority of noisy dark background and a minority distribution of 'true' signal """
    data_in_sorted = np.sort(data_in)
    Npix = len(data_in)
    pix_ax = np.arange(0, Npix, 1)
    Ipix = np.max(data_in)
    data_in_sorted = data_in_sorted / Ipix * Npix

    # fit on lower half of N:
    lowerhalf_N = data_in_sorted[0 : int(Npix / 2)]
    lowerhalf_pix_ax = pix_ax[0 : int(Npix / 2)]
    lowerfit_p = np.polyfit(lowerhalf_pix_ax, lowerhalf_N, 1)
    lowerfit = np.polyval(lowerfit_p, pix_ax)

    # fit on higher half of I:
    upperhalf_I = data_in_sorted[data_in_sorted > Npix / 2]
    upperhalf_pix_ax = pix_ax[data_in_sorted > Npix / 2]
    upperfit_p = np.polyfit(upperhalf_pix_ax, upperhalf_I, 1)
    upperfit = np.polyval(upperfit_p, pix_ax)
    # get cross-point
    xc = (lowerfit_p[1] - upperfit_p[1]) / (upperfit_p[0] - lowerfit_p[0])
    yc = np.polyval(lowerfit_p, xc)

    # get 'knee'
    rr = np.hypot(pix_ax - xc, data_in_sorted - yc)
    x_kn = pix_ax[(rr == min(rr))]
    y_kn = data_in_sorted[(rr == min(rr))]
    # scale value back
    treshold = y_kn / Npix * Ipix
    if len(treshold)>1:
        treshold =float('nan')
    return treshold

def get_nearby(x0=1, y0=1, xx=1, yy=1, r0=1):
    """
    Find selection of points near a given point, in order of distance
    distance is also exported.
    """
    rn = np.hypot(xx - x0, yy - y0)
    ix = np.argwhere(rn < r0)
    xx_near = xx[ix[:, 0]]
    yy_near = yy[ix[:, 0]]
    rn_near = rn[ix[:, 0]]
    rn_near, xx_near, yy_near = zip(*sorted(zip(rn_near, xx_near, yy_near)))

    return xx_near, yy_near, rn_near

def edge_count(bin_im):
    #1) measure the number of pixels on the edge
    #2) by pair-waise distance measurment, make an estimate of their physical contour length
    # note: isolated pixels are excluded, fragmentation is allowed  
    shrink_im = binary_erosion(bin_im, iterations = 1)
    edge_im=np.logical_and(bin_im, ~shrink_im)
    N_edge=(len(np.argwhere(edge_im>0)))
    #2) estimate the pixel distances
    #fetch coordinates
    rr, cc = bin_im.shape
    #y, x = np.indices((rr, cc))
    xy_white = np.argwhere(edge_im == 1)
    dd=[]
    for ii in range(len(xy_white)):
        x = xy_white[ii, 0]
        y = xy_white[ii, 1]
        xx_near, yy_near, rn_near=get_nearby(x,y, xy_white[:, 0], xy_white[:, 1], r0=5)
        #count 2 nearest distances, excluding self:
        if len(xx_near)>2:  #skip isolated points
            dd.append(np.sum(rn_near[1:3]))
    L_contour=np.sum(dd)/2
    dum=1
    return N_edge, L_contour

def binary_disk(rr,cc,radius):
    """ create a binary image of a disk   """
    vx, vy = np.linspace(-rr/2, rr/2, rr), np.linspace(-cc/2, cc/2, cc)
    XX, YY = np.meshgrid(vy, vx)
    radii = np.hypot(XX, YY)
    disk_im=0*radii
    disk_im[radii<radius]=1   
    return disk_im

def find_perimeter_ratio(im,Rg, pix2um,pr_):
    """ aim: get a signature of how corrugated a pattern is.
    #approach:
    # 1) raise a treshold until the area of the resulting binary image matches that of disk with equivalent Rg
    # 2) count number of pixels on edge
    # 3) scale this on the area of the equivalent disk
    # 4) interpretation: 0 is very disk-like, 1 is a cloud of isolated pixels
    #Jacob Kers 2024
     """
    im=cv2.blur(im, (3, 3))  #(used for masking)
    #a closed disk has a radius of sqrt(2) times its Rg:
    Rg_pix=Rg/pix2um[0]
    R_disk=Rg_pix*(2**0.5)
    area_limit=mt.pi*(R_disk)**2
    #find lowest (signal limit) treshold, set range to try:
    tres0=np.min(im) #get_treshold_data(np.ndarray.flatten(im))
    tres1=np.max(im)
    tresrange=np.linspace(tres0,tres1,30)
    areas=[]
    for tres in tresrange:
        bin_im=get_dna_mask(im, tres)
        ar=len(np.argwhere(bin_im>0))
        areas.append(ar)
    #mni=np.nonzero(abs(areas-area_limit)==min(abs(areas-area_limit)))
    mni = np.argmin(abs(areas-area_limit))
    #build mask with area equivalent to disk:
    bin_im=get_dna_mask(im, tresrange[mni])
    N_perim_dna, L_perim_dna=edge_count(bin_im)
    #print(R_disk[0]*2*mt.pi, N_perim_dna, L_perim_dna)
    rr, cc = np.shape(bin_im)
    # build equivalent disk for reference count:
    disk_im=binary_disk(rr,cc,R_disk)
    N_perim_disk, L_perim_disk=edge_count(disk_im)
    #print(R_disk[0]*2*mt.pi, N_perim_disk, L_perim_disk)
    N_disk=(len(np.argwhere(disk_im>0)))      

    #ratio: limit 0 means disk-like (except for the discretization limit), 1 is point cloud
    if N_disk>0:
        perimeter_ratio_L=L_perim_dna/L_perim_disk
        perimeter_ratio_N=N_perim_dna/N_perim_disk
    else:
        perimeter_ratio_L=0
        perimeter_ratio_N=0
    
    return perimeter_ratio_L, perimeter_ratio_N, bin_im

def parametrize_histogram(im, cut):
    """ obtain histogram and hi-lo percentages from an image """
    if np.max(im)>0:
        tres0=get_treshold_data(np.ndarray.flatten(im))
        im=cv2.blur(im, (2, 2)) 
        collected_pixels=(np.array(im[np.nonzero(im>tres0)]))
    else:
        im=cv2.blur(im, (2, 2))
        collected_pixels=(np.array(im))
    nbins=40
    minbin=250
    maxbin=2000
    binax = np.linspace(minbin, maxbin, nbins)
    midax=binax[0:-1]+np.diff(binax)
    pixels_hist_thisframe, edges = np.histogram(collected_pixels, binax)
    #primitive cutter (same over whole range)
    if len(collected_pixels)>0:
        loperc=100*len(np.argwhere(collected_pixels<cut))/len(collected_pixels)
        hiperc=100*len(np.argwhere(collected_pixels>=cut))/len(collected_pixels)
    else:
        loperc=0
        hiperc=0
    return midax,pixels_hist_thisframe, loperc, hiperc