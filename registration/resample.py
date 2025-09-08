"""
This script is to resample the moving image stack to the fixed image stack according to a saved transformation.
"""
import os
import SimpleITK as sitk
import loading
import skimage.io as skio
import helper
import numpy as np
import math
import resample_config as config
from skimage import draw
import glob

def prepare_data_dict(registration_cmpts):
    fixed_info = {}
    fixed_info['res'] = float(registration_cmpts['low_res'].replace('um', '')) / 1000.0  # in mm
    fixed_info['cmpt'] =[int(math.floor(co)) for co in registration_cmpts['low_res_cmpt']]
    moving_info = {}
    moving_info['res'] = float(registration_cmpts['high_res'].replace('um', '')) / 1000.0  # in mm
    moving_info['cmpt'] = [int(math.floor(co)) for co in registration_cmpts['high_res_cmpt']]
    return fixed_info, moving_info

def generate_circular_mask(diameter, slices=1):
    '''
    This function generates a circular mask for HiP-CT high resolution data.
    Args:
        radius: int, radius of the circle
        slices: int, number of slices
    '''
    radius = diameter // 2
    mask = np.ones((diameter, diameter), dtype=np.uint8)
    rr, cc = draw.disk((radius, radius), radius)
    mask[rr,cc] = 0
    if slices > 1:
        mask = np.dstack([mask]*slices)
    return mask

def crop_roi(moving_shape):
    w = moving_shape[0]
    h = moving_shape[2]
    w_new = np.fix(np.sqrt(np.square(w)/2)).astype(int)
    d = int((w - w_new) // 2)
    img_crop_coord = [(d, d, 0), (int(d+w_new), int(d+w_new), h)] # [top-left, right-bottom] -> [(x1', y1'), (x2', y2')]
    return img_crop_coord

def calculate_origin(moving_shape, moving_res):
    w = moving_shape[0]
    w_new = np.fix(np.sqrt(np.square(w)/2)).astype(int)
    d = (w - w_new) // 2
    # in mm
    d = float(d) * moving_res
    return -d

def resample_image(fix_image, moving_image, transform, default_value=0.0):
    print('\n')
    print('Resampling moving image to fixed image...')
    interpolator = sitk.sitkNearestNeighbor #sitk.sitkCosineWindowedSinc
    #skio.imsave('moving_label_before_reg.tif', sitk.GetArrayFromImage(moving_image), plugin='tifffile', check_contrast=False)
    #resampled = sitk.Resample(moving_image, fix_image, transform)
    resampled = sitk.Resample(moving_image, fix_image, transform, interpolator, default_value)
    #skio.imsave('moving_label_after_reg.tif', sitk.GetArrayFromImage(resampled), plugin='tifffile', check_contrast=False)
    #skio.imsave('fixed_image.tif', sitk.GetArrayFromImage(fix_image), plugin='tifffile', check_contrast=False)
    return resampled

def save_results(resampled_image, save_path):
    print('\n')
    print('Saving resampled image to {}'.format(save_path))
    print('Resampled image size: {}'.format(resampled_image.GetSize()))
    skio.imsave(save_path, sitk.GetArrayFromImage(resampled_image), plugin='tifffile', check_contrast=False)

def prepare_data(fixed_image_path, moving_image_path, fixed_res, moving_res, fix_crop_coords, moving_crop_coords, registration_cmpts, resample_type, moving_square_crop):
    
    fixed_res /= 1000.0
    moving_res /= 1000.0 

    print('Registering moving {} at resolution {} (mm) to fixed image at resolution {} (mm)...'.format(resample_type, moving_res, fixed_res))
    print('\n')

    fixed_info, moving_info = prepare_data_dict(registration_cmpts)

    #temp_path = transform_path.replace('_crop', '')
    #slice_nums = [int(temp_path.split('_')[-2]), int(temp_path.split('_')[-1].split('.')[0])]
    
    # loading moving image/label (high res)
    if resample_type == 'image':
        print('Loading moving images...')
        if moving_square_crop:
            moving_shape = skio.imread(glob.glob(moving_image_path+'/*.tif')[0]).shape
            z = len(glob.glob(moving_image_path+'/*.tif'))
            moving_shape = (moving_shape[0], moving_shape[1], z)
            crop_roi_coords = crop_roi(moving_shape)
            moving_im_arr = loading.loading_by_dask(moving_image_path, crop_coords=crop_roi_coords)
            #print('Moving image shape: {}'.format(moving_im_arr.shape))
        else:    
            moving_im_arr = loading.loading_by_dask(moving_image_path)
            #print('Moving image shape: {}'.format(moving_im_arr.shape))
        moving_im_arr = sitk.GetImageFromArray(moving_im_arr)
        moving_im_arr = sitk.Cast(moving_im_arr, sitk.sitkFloat32)
    elif resample_type == 'label':
        print('Loading moving labels...')
        if moving_square_crop:
            #moving_shape = skio.imread(glob.glob(moving_image_path+'/*.tif')[0]).shape
            #z = len(glob.glob(moving_image_path+'/*.tif'))
            #moving_shape = (moving_shape[0], moving_shape[1], z)
            #moving_shape = (1928, 1928, 1363)
            lbl_shape = loading.loading_by_dask(moving_image_path).shape
            moving_shape = (lbl_shape[1], lbl_shape[2], lbl_shape[0])
            crop_roi_coords = crop_roi(moving_shape)
            moving_im_arr = loading.loading_by_dask(moving_image_path, crop_coords=crop_roi_coords)
        else:
            moving_im_arr = loading.loading_by_dask(moving_image_path)
        moving_im_arr = sitk.GetImageFromArray(moving_im_arr)
        moving_im_arr = sitk.Cast(moving_im_arr, sitk.sitkUInt8)
        print('Label max: {}, min: {}'.format(sitk.GetArrayFromImage(moving_im_arr).max(), sitk.GetArrayFromImage(moving_im_arr).min()))
    else:
        raise ValueError('Invalid resample type. Please choose from "image" or "label".')
    print('Moving image size: {}'.format(moving_im_arr.GetSize()))
    
    # loading fixed image (low res)
    moving_images_num = moving_im_arr.GetSize()[2]
    fixed_images = glob.glob(fixed_image_path+'/*.tif')
    fixed_images_num = len(fixed_images)
    fixed_shape = skio.imread(fixed_images[0]).shape # (y, x)
    print('Fixed image shape: {}'.format(fixed_shape))
    #slice_nums = helper.crop_fixed_z(fixed_info, moving_info, fixed_images_num, moving_images_num)
    slice_nums = [0, fixed_images_num]
    print('Loading fixed images from slice {} to slice {}:'.format(slice_nums[0], slice_nums[1]))
    fixed_crop_coords = [(0, 0, slice_nums[0]), (fixed_shape[1], fixed_shape[0], slice_nums[1])]
    fixed_im_arr = loading.loading_by_dask(fixed_image_path, crop_coords=fixed_crop_coords)
    fixed_im_arr = sitk.GetImageFromArray(fixed_im_arr)
    print('Fixed image size: {}'.format(fixed_im_arr.GetSize()))


    # Setting origins and spacing
    print('\nRe-calculating origin...')
    if not moving_square_crop:
        d = calculate_origin(moving_im_arr.GetSize(), moving_res)
    else:
        d = 0.0
    moving_z_shift = moving_crop_coords[0][2]
    moving_z_shift = float(moving_z_shift) * moving_res 
    moving_im_arr.SetOrigin([d, d, -moving_z_shift])
    moving_im_arr.SetSpacing([moving_res, moving_res, moving_res])
    print('New moving origin: {}'.format(moving_im_arr.GetOrigin()))

    fixed_x_shift = float(fix_crop_coords[0][0]) * fixed_res
    fixed_y_shift = float(fix_crop_coords[0][1]) * fixed_res
    fixed_z_shift = fix_crop_coords[0][2] - slice_nums[0]
    fixed_z_shift = float(fixed_z_shift) * fixed_res
    fixed_im_arr.SetOrigin([-fixed_x_shift, -fixed_y_shift, -fixed_z_shift])
    fixed_im_arr.SetSpacing([fixed_res, fixed_res, fixed_res])
    #fixed_im_arr = normaliser.Execute(fixed_im_arr)
    fixed_im_arr = sitk.Cast(fixed_im_arr, sitk.sitkFloat32)
    print('New fixed origin: {}'.format(fixed_im_arr.GetOrigin()))

    return fixed_im_arr, moving_im_arr, fixed_crop_coords


if __name__ == "__main__":
    fixed_image_path = config.FIXED_IM_PATH
    resample_type = config.R_TYPE
    moving_image_path = config.MOVING_IM_PATH
    transform_path = config.T_PATH
    save_path = config.SAVE_PATH
    fixed_res = config.FIXED_RES
    moving_res = config.MOVING_RES
    registration_list_json = config.REGISTRATION_LIST
    registration_res = config.REGISTRATION_RES
    moving_square_crop = config.MOVING_SQAURE_CROP

    #temp_path = transform_path.replace('_crop', '')
    # open tfm file
    t_file = glob.glob(os.path.join(transform_path, '*.tfm'))[0]
    print('Reading transformation from {}'.format(t_file))
    transform = sitk.ReadTransform(t_file)
    print('\n')
    print('Transformation: {}'.format(transform))
    print('\n')

    registration_cmpts = helper.read_cmpt_file(registration_list_json)
    registration_cmpts = registration_cmpts[registration_res]
    register_info_json_files = glob.glob(os.path.join(transform_path, '*.json'))
    print('Register info files: {}'.format(register_info_json_files))
    for json_file in register_info_json_files:
        if 'fixed' in json_file:
            fixed_crop_coords = helper.read_cmpt_file(json_file)['crop_coords']
        elif 'moving' in json_file:
            moving_crop_coords = helper.read_cmpt_file(json_file)['crop_coords']
    # print('Fixed info from reg: {}'.format(fixed_info_from_reg))
    # print('Moving info from reg: {}'.format(moving_info_from_reg))
    fixed_im_arr, moving_im_arr, fixed_crop_coords = prepare_data(fixed_image_path, 
                                                             moving_image_path, 
                                                             fixed_res, 
                                                             moving_res, 
                                                             fixed_crop_coords,
                                                             moving_crop_coords,
                                                             registration_cmpts, 
                                                             resample_type,
                                                             moving_square_crop)
    fixed_crop_z = [fixed_crop_coords[0][2], fixed_crop_coords[1][2]]
    resampled_save_path = os.path.join(save_path, str(moving_res) + '_to_' + str(fixed_res) +'_resample_' + resample_type + '_slice_' + str(fixed_crop_z[0]) + '_' + str(fixed_crop_z[1]) + '.tif')
    
    if resample_type == 'image':
       resampled_image = resample_image(fixed_im_arr, moving_im_arr, transform, default_value=255.0)
       resampled_image = sitk.Cast(resampled_image, sitk.sitkUInt8)
    else:
       resampled_image = resample_image(fixed_im_arr, moving_im_arr, transform, default_value=0.0)
       resampled_image = sitk.Cast(resampled_image, sitk.sitkUInt8)
    
    save_results(resampled_image, resampled_save_path)

    if config.SAVE_IMAGE:
        save_path = os.path.join(save_path, str(fixed_res) + '_slice_' + str(fixed_crop_z[0]) + '_' + str(fixed_crop_z[1]) + '.tif')
        fixed_im_arr = loading.loading_by_dask(fixed_image_path, crop_coords=fixed_crop_coords)
        print('Fixed image size: {}'.format(fixed_im_arr.shape))
        print('Saving fixed image to {}'.format(save_path))
        skio.imsave(save_path, fixed_im_arr, plugin='tifffile', check_contrast=False)