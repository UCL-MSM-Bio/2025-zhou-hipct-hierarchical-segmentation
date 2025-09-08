"""
HiP-CT Registration Helper Functions
"""
import os
import numpy as np
import skimage.io as skio
import json
import glob
import tqdm
import natsort
import SimpleITK as sitk

def convert_16_to_8(im_itk_arr):
    """
    This function converts 16-bits itk stack to 8-bits itk stack
    Args:
        im_itk_arr: 16-bits itk stack
    Returns:
        im_sitk_8: 8-bits itk stack
    """
    im_arr_16 = im_itk_arr
    im_sitk_8 = sitk.Cast(sitk.RescaleIntensity(im_arr_16), sitk.sitkUInt8)
    return im_sitk_8

def crop_fixed_roi(fixed_info, moving_info, moving_shape, fix_shape, R_FACTOR):
    # fix_shape, moving_shape = (y,x,z)
    (cmpt_x, cmpt_y, cmpt_z) = fixed_info['cmpt']
    #print('fixed_shaoe:', fix_shape)
    print('original cmpt:', cmpt_x, cmpt_y, cmpt_z)
    x_left_bound = cmpt_x - R_FACTOR * (moving_info['cmpt'][0] * moving_info['res'] / fixed_info['res'])
    x_left_bound = int(x_left_bound)
    x_left_bound = max(0, x_left_bound)
    x_right_bound = cmpt_x + R_FACTOR * ((moving_shape[1] - moving_info['cmpt'][0]) * moving_info['res'] / fixed_info['res'])
    x_right_bound = int(x_right_bound)
    x_right_bound = min(fix_shape[1], x_right_bound)
    cmpt_x_new = int(cmpt_x - x_left_bound)
    print('x_left_bound:', x_left_bound)
    print('x_right_bound:', x_right_bound)
    print('cmpt_x_new:', cmpt_x_new)

    y_lower_bound = cmpt_y - R_FACTOR * (moving_info['cmpt'][1] * moving_info['res'] / fixed_info['res'])
    y_lower_bound = int(y_lower_bound)
    y_lower_bound = max(0, y_lower_bound)
    y_upper_bound = cmpt_y + R_FACTOR * ((moving_shape[0] - moving_info['cmpt'][1]) * moving_info['res'] / fixed_info['res'])
    y_upper_bound = int(y_upper_bound)
    y_upper_bound = min(fix_shape[0], y_upper_bound)
    cmpt_y_new = int(cmpt_y - y_lower_bound)
    print('y_lower_bound:', y_lower_bound)
    print('y_upper_bound:', y_upper_bound)
    print('cmpt_y_new:', cmpt_y_new)

    z_lower_bound = cmpt_z - R_FACTOR * (moving_info['cmpt'][2] * moving_info['res'] / fixed_info['res'])
    z_lower_bound = int(z_lower_bound)
    z_lower_bound = max(0, z_lower_bound)
    z_upper_bound = cmpt_z + R_FACTOR * ((moving_shape[2] - moving_info['cmpt'][2]) * moving_info['res'] / fixed_info['res'])
    z_upper_bound = int(z_upper_bound)
    z_upper_bound = min(fix_shape[2], z_upper_bound)
    cmpt_z_new = int(cmpt_z - z_lower_bound)
    print('z_lower_bound:', z_lower_bound)
    print('z_upper_bound:', z_upper_bound)
    print('cmpt_z_new:', cmpt_z_new)

    img_crop_coord = [(x_left_bound, y_lower_bound, z_lower_bound), (x_right_bound, y_upper_bound, z_upper_bound)] # [top-left, right-bottom] -> [(x1', y1'), (x2', y2')]
    return img_crop_coord, (cmpt_x_new, cmpt_y_new, cmpt_z_new)

def crop_moving_roi(moving_shape, moving_info, z_factor=1e6):
    (cmpt_x, cmpt_y, cmpt_z) = moving_info['cmpt']
    w = moving_shape[0]
    h = moving_shape[2]
    # w_new is the width of the biggest square in inside the circle
    w_new = np.fix(np.sqrt(np.square(w)/2)).astype(int)
    d = int((w - w_new) // 2)
    (cmpt_x_new, cmpt_y_new) = (cmpt_x - d, cmpt_y - d)
    
    # z_factor is a factor to determine the z range to crop, default 0.5. 
    # if you want to take the whole z range, set z_factor to a large number, e.g. 1e6
    z_start = int(cmpt_z - z_factor * w_new) # 0.5 can be an input parameter in the future
    z_end = int(cmpt_z + z_factor * w_new)
    z_start = max(0, z_start)
    z_end = min(h, z_end)
    cmpt_z_new = cmpt_z - z_start
    img_crop_coord = [(d, d, z_start), (int(d+w_new), int(d+w_new), int(z_end))] # [top-left, right-bottom] -> [(x1', y1'), (x2', y2')]
    return img_crop_coord, (cmpt_x_new, cmpt_y_new, cmpt_z_new)

def read_cmpt_file(cmpt_file):
    """
    File is in json format
    """
    file_type = cmpt_file.split('.')[-1]
    if file_type != 'json':
        raise ValueError('File type is not json')
    with open(cmpt_file, 'r') as cf:
        data = json.load(cf)
    return data

def check_integrity(registration_cmpts, fixed_image_folder, moving_image_folder):
    '''
    Check if the registration resolution matches the paths of fixed and moving images 
    '''
    fixed_res_from_cmpt = registration_cmpts['low_res']
    moving_res_from_cmpt = registration_cmpts['high_res']
    if not fixed_res_from_cmpt in fixed_image_folder:
        raise ValueError('Fixed image path does not match registration resolution')
    if not moving_res_from_cmpt in moving_image_folder:
        raise ValueError('Moving image path does not match registration resolution')

def jp2_to_tif(jp2_file_path, tif_save_path):
    '''
    HiP-CT images are in jp2, but we will process it with tif 
        as we use dask_image package
    Args:
        jp2_file_path: path to jp2 file
        tif_save_path: path to save tif file
    '''
    jp2_files = glob.glob(jp2_file_path)
    for jp2_file in tqdm.tqdm(jp2_files):
        image = skio.imread(jp2_file)
        file_name = jp2_file.split('/')[-1]
        file_name = file_name.replace('.jp2', '.tif')
        tif_save_path = os.path.join(tif_save_path, file_name)
        skio.imsave(tif_save_path, image, plugin='tifffile', check_contrast=False)

def crop_fixed_z(fixed_info, moving_info, fixed_images_num, moving_images_num, R_FACTOR=1.1):
    z_lower_bound = fixed_info['cmpt'][2] - R_FACTOR * (moving_info['cmpt'][2] * moving_info['res'] / fixed_info['res'])
    z_lower_bound = int(z_lower_bound)
    z_lower_bound = max(0, z_lower_bound)
    z_upper_bound = fixed_info['cmpt'][2] + R_FACTOR * ((moving_images_num - moving_info['cmpt'][2]) * moving_info['res'] / fixed_info['res'])
    z_upper_bound = int(z_upper_bound)
    z_upper_bound = min(fixed_images_num, z_upper_bound)
    return (z_lower_bound, z_upper_bound)