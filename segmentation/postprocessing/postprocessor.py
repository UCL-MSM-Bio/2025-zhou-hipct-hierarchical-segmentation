import io
import os
import glob
import tqdm
import skimage.io as skio
from natsort import natsorted
import numpy as np
from tqdm import tqdm


def divide_subvolumes(big_vol_path, segments=2, overlap_depth_in_pixels=0):
    '''
    This function divides a big volume into subvolumes.
    Args:
        big_vol_path: str, path to the big volume
        segments: int, number of segments to divide the volume into
        overlap_depth_in_pixels: int, overlap depth in pixels
    Returns:
        None
    '''
    big_vol = skio.imread(big_vol_path, plugin='tifffile')
    print(f'Big volume shape: {big_vol.shape}')
    vol_depth = big_vol.shape[0]

    segment_size = vol_depth // segments
    print(f'Segment size: {segment_size}')

    if overlap_depth_in_pixels >= segment_size:
        raise ValueError(f'Overlap depth {overlap_depth_in_pixels} is greater than or equal to segment size {segment_size}. Please reduce the overlap depth.')
    
    start = 0
    end = 0
    for i in tqdm.tqdm(range(segments)):
        start = end - overlap_depth_in_pixels if i > 0 else 0
        end = start + segment_size if i < segments - 1 else vol_depth-1
        
        if start < 0:
            start = 0
        if end > vol_depth:
            end = vol_depth
        
        subvol = big_vol[start:end]
        save_path = os.path.join(os.path.dirname(big_vol_path), os.path.basename(big_vol_path).replace('.tif', f'_subvol_{start}_{end}.tif'))
        skio.imsave(save_path, subvol, plugin='tifffile', check_contrast=False)
        print(f'Subvolume saved at: {save_path}')
        print(f'Subvolume shape: {subvol.shape}')
    print('Subvolume division completed.')

def concatenate_subvolumes(inference_dir, overlap_depth_in_pixels=0, output_path=None):
    '''
    This function concatenates subvolumes into a big volume.
    Args:
        subvolumes_path: str, path to the directory containing subvolumes
        output_path: str, path to save the concatenated volume
    Returns:
        None
    '''
    subvol_folders = [f.path for f in os.scandir(inference_dir) if f.is_dir()]
    subvol_folders = natsorted(subvol_folders)
    if len(subvol_folders) == 0:
        raise ValueError(f'No subvolume folders found in {inference_dir}. Please check the path.')
    print(f'Found {len(subvol_folders)} subvolume folders.')
    subvol_files = []
    for folder in subvol_folders:
        files = glob.glob(os.path.join(folder, '*.tif'))
        if len(files) == 0:
            raise ValueError(f'No .tif files found in {folder}. Please check the path.')
        subvol_files.extend(files)
    print(subvol_files)
    total_subvols = len(subvol_files)

    big_vol = []
    half_overlap = overlap_depth_in_pixels // 2
    segment_num = 0
    for subvol_file in tqdm.tqdm(subvol_files):
        subvol = skio.imread(subvol_file, plugin='tifffile')
        if segment_num == 0:
            if overlap_depth_in_pixels > 0 and total_subvols > 1:
                big_vol.append(subvol[:-half_overlap])
            else:
                big_vol.append(subvol)
        else:
            if overlap_depth_in_pixels > 0 and segment_num < total_subvols - 1:
                big_vol.append(subvol[half_overlap:-half_overlap])
            elif overlap_depth_in_pixels > 0 and segment_num == total_subvols - 1:
                big_vol.append(subvol[half_overlap:])
            else:
                big_vol.append(subvol)
        segment_num += 1

    big_vol = np.concatenate(big_vol, axis=0)
    skio.imsave(output_path, big_vol, plugin='tifffile', check_contrast=False)
    print(f'Concatenated volume saved at: {output_path}')
    print(f'Concatenated volume shape: {big_vol.shape}')

def remove_on_density(instance, x_center, y_center, z_center,shape=(3770, 1898, 1898), size=128, threshold=32):
    flag = False

    i_x_centre = instance['centroid_0']
    i_y_centre = instance['centroid_1']
    i_z_centre = instance['centroid_2']
    i_x_start = int(i_x_centre - size//2) if (i_x_centre - size//2) > 0 else 0
    i_y_start = int(i_y_centre - size//2) if (i_y_centre - size//2) > 0 else 0
    i_z_start = int(i_z_centre - size//2) if (i_z_centre - size//2) > 0 else 0
    i_x_end = int(i_x_centre + size//2) if (i_x_centre + size//2) <= shape[2] else shape[2]
    i_y_end = int(i_y_centre + size//2) if (i_y_centre + size//2) <= shape[1] else shape[1]
    i_z_end = int(i_z_centre + size//2) if (i_z_centre + size//2) <= shape[0] else shape[0]

    count = -1 # count itself
    for x, y, z in zip(x_center, y_center, z_center):
        if x > i_x_start and x < i_x_end: 
            if y > i_y_start and y < i_y_end:
                if z > i_z_start and z < i_z_end:
                    count += 1
                    if count >= threshold:
                        break
    if count < threshold:
        flag = True
    return flag

def apply_mask(mask_path, stack):
    mask = skio.imread(mask_path, plugin='tifffile')
    if mask.shape != stack.shape:
        raise ValueError(f'Mask shape {mask.shape} and stack shape {stack.shape} do not match.')
    masked_stack = np.multiply(stack, mask)
    masked_stack[stack > 0] = 1
    masked_stack = masked_stack.astype(np.uint8)
    return masked_stack
