"""

This file contains functions that take as argument a 3D stack of images.

Please note: format should be t,y,x, since this is much more efficient in terms of indexing!

Author: Janni Harju
Created: July 30th, 2024

"""
import multiprocessing as mp #for calculating spatial correlations
import numpy as np
import h5py
from pygbox import ops #for ops.bg_sub
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist

def bg_sub_stack(stack, mask_stack, how = "out"):
    """ Subtract background from an image stack knowing foreground masks.
    Also apply the masks
    """
    check_dims(stack,4)
    check_dims(mask_stack,4)
    nt=stack.shape[0]
    for t in range(nt):
        frames=stack[t,:,:,:]
        masks=mask_stack[t,:,:,:]
        frames=ops.bg_sub(frames, masks, how=how)
        stack[t,:,:,:]=frames*masks

    return stack

def check_dims(stack, ndims=3):
    if stack.ndim!=ndims:
        raise ValueError(f'Got an array of size {stack.shape}. Was expecting {ndims}')

def subtract_median_normalize_each_frame(stack):
    """Mainly for simulations; subtract the median value as an alternative to masking
    """
    check_dims(stack,3)
    nt=stack.shape[0]
    stack[np.isnan(stack)]=0
    for t in range(nt):
        frame=stack[t,:,:]
        bg=frame[frame!=0]
        bg_val = np.nanmedian(bg)
        frame=frame-(bg_val)
        frame[frame<0]=0

        if np.nanmax(frame)>0:
            stack[t,:,:]=frame/np.nanmax(frame)
    return stack

def center_of_mass_diffusion(stack):
    """for each time point, calculate the center of mass position.

    Then track MSD of center of mass
    Args:
        stack (3D np.array): A stack of images (t, y, x)

    Returns:
        ndarray: nt sized array of center of mass MSD over time 

    References:
        Author: Janni 
    """
    check_dims(stack,3)
    nt=stack.shape[0]
    com_pos=np.zeros((nt,2))
    for t in range(nt):
        frame=stack[t,:,:]
        if np.any(frame):
            com_pos[t,:]=center_of_mass(frame)
   
    #displacement from time zero
    for t in range(nt):
        com_pos[t,:]-=com_pos[0,:]
    
    com_msd=com_pos[:,0]**2+com_pos[:,1]**2
    return com_msd

def difference_in_time(stack):
    """ Calculate changes in the field over time 

    Args:
        stack (3D np.array): A stack of images (t, y, x)

    Returns:
        ndarray: nt-1 sized array of I[t+1]-I[t] for each pixel and time point 

    References:
        Author: Janni 
    """
    check_dims(stack,3)
    nt, ny, nx=stack.shape
    time_differences=np.zeros((nt-1,ny,nx))
    for i in range(nt-1):
        time_differences[i,:,:]=stack[i+1,:,:]-stack[i,:,:]

    return time_differences

def laplacians(stack, dx):
    """ Computes the discrete laplacians for (projected) image 

    Args:
        stack (3D np.array): A stack of images (t, y, x)
        dx : pixel size in x/y 

    Returns:
        ndarray: laplacian for (projected) image 

    References:
        Author: Janni 
    """
    check_dims(stack, 3)
    nt=stack.shape[0]
    laps=np.zeros(stack.shape)
    
    # Define the Laplacian kernel
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]])
    
    # Apply the Laplacian kernel to each frame
    for i in range(nt):
        lap=convolve2d(stack[i,:,:], kernel, mode="same")
        laps[i,:,:] = lap

    return laps/(dx**2)

def radial_profile_1d(stack, distances,trap_center, dx):
    """ calculates the mean density as a function of distance from the center of mass

    Uses a projection

    Args:
        stack (3D np.array): A stack of images (t, y, x)
        distances (ndarray): 1D array of distances used for binning
        dx (list): list containing pixelsize i.e. [dz,dy,dx]

    Returns:
        ndarray : 1D array of mean intensity at distance

    References:
        Author: Janni
    """
    check_dims(stack,3)
    check_dims(distances,1)

    nt,ny,nx=stack.shape

    intensities=np.zeros_like(distances)
    counts=np.zeros_like(distances)

    #Distances to trap center
    y_indices, x_indices = np.indices((ny,nx))
    ds = np.hypot(dx[1]*(y_indices - trap_center[0]), dx[2]*(x_indices - trap_center[1])) #in actual units
    binned_ds=np.digitize(ds, distances)-1

    for t in range(nt):
        frame=stack[t,:,:]
        for i in range(len(distances)):
            bin_mask=binned_ds==i #everything at correct distances
            if np.any(bin_mask):
                bin_intensities=frame[bin_mask]
                bin_intensities=bin_intensities[bin_intensities!=0] #exclude masked out pixels
                intensities[i]+=np.nansum(bin_intensities)
                counts[i]+=bin_intensities.size

    if np.any(counts):
        intensities[counts>0]/=counts[counts>0]
        #return the normalized array
        return intensities/np.nansum(intensities)
    else:
        return np.zeros_like(distances)

def density_profile_1d(stack, distances, dx):
    """ calculates the mean density as a function of distance from the center of mass

    Uses a projection

    Args:
        stack (3D np.array): A stack of images (t, y, x)
        distances (ndarray): 1D array of distances used for binning
        dx (list): list containing pixelsize i.e. [dz,dy,dx]

    Returns:
        ndarray : 1D array of mean intensity at distance

    References:
        Author: Janni
    """
    check_dims(stack,3)
    check_dims(distances,1)

    nt,ny,nx=stack.shape

    intensities=np.zeros_like(distances)
    counts=np.zeros_like(distances)

    for t in range(nt):
        frame=stack[t,:,:]
        if np.any(frame):
            cm = center_of_mass(frame)

            #Distances to CoM
            rows, cols = frame.shape
            y_indices, x_indices = np.indices((rows, cols))
            ds = np.hypot(dx[1]*(y_indices - cm[0]), dx[2]*(x_indices - cm[1])) #in actual units
            binned_ds=np.digitize(ds, distances)-1

            for i in range(len(distances)):
                bin_mask=binned_ds==i #everything at correct distances
                if np.any(bin_mask):
                    bin_intensities=frame[bin_mask]
                    bin_intensities=bin_intensities[bin_intensities!=0] #exclude masked out pixels
                    intensities[i]+=np.nansum(bin_intensities)
                    counts[i]+=bin_intensities.size
    
    intensities[counts>0]/=counts[counts>0]
    #return the normalized array
    return intensities/np.nansum(intensities)

#a few helper functions for calculating the mean correlation as a function of distance
def precalculate_distance_bins(ny,nx,distances, dx):
    """Given the size of a 2D array and bins of distances, calculate an array that gives distance bin numbers for pairs of points
    Args:
        ny (int): size of array in 0th dim
        nx (int): size of array in 1st dim
        distances (np.array): vector of distance bins for pairwise distances
        dx (list): list containing pixelsize i.e. [dz,dy,dx]

    Returns:
        np.array: for a pair of LINEAR indices, stores the bin number of the pixel-pixel distance

    """
    indices = np.indices((ny, nx)).reshape(2, -1).T #ny*nx,2 array. 
    distance_matrix = cdist(indices, indices, metric='euclidean') #2D array. Each dimension is a linear index.
    bin_indices = np.digitize(distance_matrix*dx[-1], distances) - 1
    return bin_indices

def find_global_bounding_box(stack):
    """Find bounding box for cropping pixels that are zero for all time points
    Args:
        stack (3D np.array): A stack of images (t, y, x)

    Returns:
        tuple: indices for the top-left corner for cropping
        tuple: indices for the bottom-right corner for cropping
    """
    non_zero_indices = np.argwhere(stack)
    if non_zero_indices.size == 0:
        return (0,0), (0,0) 
    top_left = non_zero_indices.min(axis=0)[1:]  # Considering only x, y dimensions
    bottom_right = non_zero_indices.max(axis=0)[1:]
    return top_left, bottom_right

def crop_to_global_bbox(stack):
    """Crops a 3D array so that pixels that are zero for all time points are removed
    Args:
        stack (3D np.array): A stack of images (t, y, x)

    Returns:
        np.array: a smaller stack, cropped to just fit pixels that are non-zero at some t and z 
    """
    top_left, bottom_right = find_global_bounding_box(stack) 
    cropped_stack = stack[:,top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    return cropped_stack

def process_frame(frame, bin_indices, n_distances):
    """
    Function for calculating the spatial correlation of a single frame
    """
    non_zeros = np.nonzero(frame)[0]
    intensities = np.zeros(n_distances, dtype=float)
    counts = np.zeros(n_distances, dtype=int)

    if non_zeros.size > 0:
        mean_frame=np.nanmean(frame[non_zeros])
        var_frame=np.nanvar(frame[non_zeros]) #for normalizing so 1 at t=0

        #all pairs of non-zero indices 
        ind_i, ind_j = np.triu_indices(non_zeros.size, k=0)

        #fetch the linearized indices
        pos_i=non_zeros[ind_i] 
        pos_j=non_zeros[ind_j]

        #distance bin index for each pair
        bin_index=bin_indices[pos_i, pos_j]

        #intensities for each pair
        val_i=frame[pos_i]
        val_j=frame[pos_j]

        #increment arrays
        np.add.at(intensities,bin_index,(val_i*val_j)/var_frame)
        np.add.at(counts, bin_index, 1)

    return intensities, counts

def spatial_correlation(stack, distances, dx, num_threads=8):
    """
    Calculate the spatial correlation of pixel intensities as a function of pixel-pixel distance.
    Corresponds to:
            cor(d)=<(I(x,t)-μ)(I(y,t)-μ)/σ^2 | |x-y|=d>
    where the average is taken over all pairs of pixels with given distance d, and over all time-points.
    The mean μ and the variance σ^2 are calculated for each frame.

    Args:
        stack (3D np.array): A stack of images (t, y, x)
        distances (ndarray): 1D array of distances used for binning
        dx (list): list containing pixelsize i.e. [dz,dy,dx]

    Returns:
        np.array: Spatial correlation values for each pixel-pixel distance

    References:
        Author: Janni
    """
    if num_threads>1:
        print(f"using {num_threads} cpus for spatial correlation calculations. available: {mp.cpu_count()}")

    check_dims(stack, 3)
    check_dims(distances,1)
   
    #get rid of zeros to speed things up...
    cropped_stack=crop_to_global_bbox(stack)
    print(f"original shape {stack.shape}. after crop: {cropped_stack.shape}")

    nt,ny,nx=cropped_stack.shape
    nd=distances.shape[0]

    # Calculate the pairwise distances between the indices
    bin_indices=precalculate_distance_bins(ny,nx,distances,dx)

    #use threads to calculate the spatial correlation for each frame
    with mp.Pool(num_threads) as pool:
        results=pool.starmap(process_frame,[(cropped_stack[t, :, :].flatten(), bin_indices, nd) for t in range(nt)])

    total_intensities = np.nansum([result[0] for result in results], axis=0)
    total_counts = np.nansum([result[1] for result in results], axis=0)
    
    valid_counts = total_counts > 0
    total_intensities[valid_counts] /= total_counts[valid_counts]
    
    return total_intensities

def temporal_correlation(stack, dt):
    """
    Calculate the mean temporal correlation of pixels over time.
    Corresponds to
            cor(t)=<(I(t_0+t)-μ)(I(t_0)-μ)/σ^2>
    where the average is taken over t_0 and all pixels.

    Parameters:
        stack (3D np.array): A stack of images (t, y, x)
        dt: the time step

    Returns:
        np.array: the times
        np.array: Temporal correlation values for each time lag

    References:
        Author: Janni
    """
    check_dims(stack,3)

    nt=stack.shape[0]
    mean_corr=np.zeros(nt)
    ts=np.arange(nt)*dt
    pixel_count=0
    
    # Extract time series for non-zero pixels
    for y,x in np.argwhere(stack[0,:,:]!=0):
        time_series = stack[:,y,x]
        time_series = time_series-np.nanmean(time_series)
        # Calculate the autocorrelation function
        autocorr = np.correlate(time_series, time_series, mode='full')
        #take the second half (forward correlation) and normalize by number of pairs
        autocorr = autocorr[nt-1:]/np.arange(nt,0,-1)

        mean_corr+=autocorr/np.nanvar(time_series)
        pixel_count+=1
    if pixel_count>0:
        return ts, mean_corr/pixel_count
    else:
        return ts, np.zeros(nt)

def analyze_stack(images_3d, dt, pix2um, trap_radius, trap_center, out_file_name, subtract_median=False, num_threads=8, compression="gzip"):
    check_dims(images_3d, 3) 
    if subtract_median: #mainly for simulations
        images_3d=subtract_median_normalize_each_frame(images_3d)

    if not np.any(images_3d):
        print("got all zeros for stack, skipping")
        return

    nt=images_3d.shape[0]
    with h5py.File(out_file_name, 'w') as out_file:
        out_file.attrs['dt']=dt
        out_file.attrs['trap_center']=trap_center

        #calculate laplacians and differences in time for comparison
        dif_in_time=difference_in_time(images_3d)
        out_file.create_dataset("differences_time", data=dif_in_time, compression=compression)

        laps=laplacians(images_3d, pix2um[-1])
        out_file.create_dataset("laplacians", data=laps)

        #as a control, we also shuffle the images and do a time difference
        shuffled_inds=np.random.permutation(nt-1)
        shuffled_dif_in_time=difference_in_time(images_3d[shuffled_inds])
        out_file.create_dataset("shuffled_differences_time",data=shuffled_dif_in_time, compression=compression)

        #mean density as function of distance from trap center
        ds_stack=np.arange(trap_radius)*pix2um[-1] #max distance just one radius now
        radial_intensities=radial_profile_1d(images_3d, ds_stack, trap_center, pix2um)
        out_file.create_dataset("radial_profile_distances", data=ds_stack, compression=compression)
        out_file.create_dataset("radial_profile", data=radial_intensities, compression=compression)

        #mean density at given distance from center of mass
        ds_stack=np.arange(2*trap_radius)*pix2um[-1] #max distance 2 radii
        intensities_stack=density_profile_1d(images_3d, ds_stack, pix2um)  #max distance 2 radii
        out_file.create_dataset("density_profile_distances", data=ds_stack, compression=compression)
        out_file.create_dataset("density_profile", data=intensities_stack, compression=compression)

        #mean pixel correlation in time
        ts, t_correlation=temporal_correlation(images_3d, dt)
        out_file.create_dataset("time_correlation", data=t_correlation, compression=compression)
        out_file.create_dataset("ts_correlation", data=ts, compression=compression)

        #heaviest calculation; correlation as function of pixel-pixel distance
        d_correlation=spatial_correlation(images_3d, ds_stack, pix2um, num_threads)
        out_file.create_dataset("distance_correlation", data=d_correlation, compression=compression)
