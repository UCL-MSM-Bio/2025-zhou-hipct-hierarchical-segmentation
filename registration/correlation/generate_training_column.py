import os 
import numpy as np
import skimage.io as skio
import dask_image.imread

COORDS = {
    'LADAF-2020-27':{
        '2.58um_to_12.1um':{
            "x_range": (765, 1049), # last already + 1
            "y_range": (819, 1103), # last already + 1
            "z_range": (123, 1281), # last already + 1
            "crop_z_range": (2371, 3769)
        },
        '12.1um_to_25.08um':{
            "x_range": (1149, 1806), # already + 1
            "y_range": (1349, 2006), # already + 1
            "z_range": (120, 1968), # already + 1
            "crop_z_range": (776, 2892)
        }
    },
    'LADAF-2021-17':{
        '5.2um_to_13um_voi_2-1':{
            "x_range": (890, 1434), # last already + 1
            "y_range": (615, 1159), # last already + 1
            "z_range": (761, 2154), # last already + 1
            "crop_z_range": (0, 0)
        },
        '5.2um_to_13um_voi_3-1':{
            "x_range": (879, 1423), # last already + 1
            "y_range": (876, 1420), # last already + 1
            "z_range": (1918, 2460), # last already + 1
            "crop_z_range": (0, 0)
        },
        '13um_to_25um_voi_2':{
            "x_range": (546, 972), # last already + 1
            "y_range": (1726, 2152), # last already + 1
            "z_range": (742, 1900), # last already + 1
            "crop_z_range": (0, 0)
        },
        '13um_to_25um_voi_3':{
            "x_range": (135,547), # last already + 1
            "y_range": (885,1218), # last already + 1
            "z_range": (138, 3244), # last already + 1
            "crop_z_range": (0, 0)
        }
    }
}

def process_image_by_dask(image_folder, x_range, y_range, z_range=None):
    '''
    Load images by dask_image.imread.imread
    Args:
        image_folder: path to image folder
        crop_z_range: crop z range (only for fixed image)
        moving_crop_coords: crop coordinates for moving image ROI (only for moving image) [top_left, bottom_right]
    '''
    image_stack = dask_image.imread.imread(f'{image_folder}/*.tif') # [slice, h, w]
    image_stack = image_stack[:,y_range[0]:y_range[1], x_range[0]:x_range[1]]
    if z_range is not None:
        image_stack = image_stack[z_range[0]:z_range[1],:,:]
    image_array = image_stack.compute()
    return image_array 

def process_label_by_dask(label_path,x_range, y_range, z_range=None):
    '''
    Load images by dask_image.imread.imread
    Args:
        image_folder: path to image folder
        crop_z_range: crop z range (only for fixed image)
        moving_crop_coords: crop coordinates for moving image ROI (only for moving image) [top_left, bottom_right]
    '''
    label_stack = dask_image.imread.imread(label_path) # [slice, h, w]
    label_stack = label_stack[:,y_range[0]:y_range[1], x_range[0]:x_range[1]]
    if z_range is not None:
        label_stack = label_stack[z_range[0]:z_range[1],:,:]
    label_array = label_stack.compute()
    return label_array

def crop_low_res_roi(low_res_dir, registered_lbl_path, save_path, coordinates):
    """
    Crop the registered ROI from the low resolution image
    Args:
        low_res_column: low resolution image, 8bit, tif
        registered_column: registered image, 8bit, tif
    """
    print('Generateing multi-resolution dataset...')
    x_range = coordinates['x_range']
    y_range = coordinates['y_range']
    z_range = coordinates['z_range']
    crop_z_range = (coordinates['crop_z_range'][0]+z_range[0], coordinates['crop_z_range'][0]+z_range[1])
    print('Cropping x range:', x_range)
    print('Cropping y range:', y_range)
    print('Cropping z range for label:', z_range)
    print('Cropping z range for image:', crop_z_range)
    
    print('\nProcessing images...')
    low_res_vol = process_image_by_dask(low_res_dir, x_range, y_range, crop_z_range)
    print('Processing labels...')
    registered_lbl_vol = process_label_by_dask(registered_lbl_path, x_range, y_range, z_range)
    print(f'Images shape:{low_res_vol.shape}; Labels shape:{registered_lbl_vol.shape}')
    print(f'Images dtype:{low_res_vol.dtype}; Labels dtype:{registered_lbl_vol.dtype}')
    print(f'Label max:{registered_lbl_vol.max()}; Label min:{registered_lbl_vol.min()}')

    print('\nSaving...')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    skio.imsave(os.path.join(save_path, 'low_res_vol_roi.tif'), low_res_vol, plugin='tifffile', check_contrast=False)
    skio.imsave(os.path.join(save_path, 'registered_lbl_roi.tif'), registered_lbl_vol, plugin='tifffile', check_contrast=False)
    
if __name__ == '__main__':
    low_res_vol_path = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/clahe'
    registered_lbl_path = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/registered/12.1_to_25.08_resample_label.tif'
    save_path = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/training/training_vol_roi'
    coordinates = COORDS['LADAF-2020-27']['12.1um_to_25.08um']
    crop_low_res_roi(low_res_vol_path, registered_lbl_path, save_path, coordinates)
