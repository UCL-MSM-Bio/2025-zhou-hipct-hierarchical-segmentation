import os
import numpy as np
import skimage.io as io
import glob
from natsort import natsorted
import json
import tqdm
from helper import reading_data
import shutil

def extract_cubes_for_hyperparams_search(im_stack, cube_dict, save_dir, padding=32):
    im = im_stack
    max_z, max_y, max_x = im.shape
    save_path = save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for cube_name in cube_dict:
        cube_info = cube_dict[cube_name]
        print('Extracting cube: ', cube_name)
        l_z, u_z = cube_info['z_range'] # lower, upper bounds
        l_y, u_y = cube_info['y_range']
        l_x, u_x = cube_info['x_range']
        shape_before_padding = (u_z - l_z, u_y - l_y, u_x - l_x)
        print('Shape before padding: ', shape_before_padding)
        l_z = 0 if (l_z-padding) < 0 else (l_z-padding)
        l_y = 0 if (l_y-padding) < 0 else (l_y-padding)
        l_x = 0 if (l_x-padding) < 0 else (l_x-padding) 
        u_z = max_z if (u_z+padding) > max_z else (u_z+padding)
        u_y = max_y if (u_y+padding) > max_y else (u_y+padding)
        u_x = max_x if (u_x+padding) > max_x else (u_x+padding)
        cube = im[l_z:u_z, l_y:u_y, l_x:u_x]
        print('Shape after padding: ', cube.shape)
        io.imsave(os.path.join(save_path, cube_name+'.tif'), cube, plugin='tifffile', check_contrast=False)
        if 'empty' in cube_name:
            mask = np.zeros_like(np.random.rand(*shape_before_padding))
            print('Mask shape: ', mask.shape)
            mask_save_path = os.path.join(os.path.dirname(save_path), 'search_gt')
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
            io.imsave(os.path.join(mask_save_path, cube_name+'_gt.tif'), mask, plugin='tifffile', check_contrast=False)

def extract_preds_for_hyperparams_search(pred_stack, cube_dict, save_path, padding=32):
    pred = pred_stack
    max_z, max_y, max_x = pred.shape
    #save_path = os.path.join(save_path, 'preds')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for cube_name in cube_dict:
        cube_info = cube_dict[cube_name]
        print('Extracting cube: ', cube_name)
        l_z, u_z = cube_info['z_range'] # lower, upper bounds
        l_y, u_y = cube_info['y_range']
        l_x, u_x = cube_info['x_range']

        l_z = 0 if (l_z-padding) < 0 else (l_z-padding)
        l_y = 0 if (l_y-padding) < 0 else (l_y-padding)
        l_x = 0 if (l_x-padding) < 0 else (l_x-padding) 
        u_z = max_z if (u_z+padding) > max_z else (u_z+padding)
        u_y = max_y if (u_y+padding) > max_y else (u_y+padding)
        u_x = max_x if (u_x+padding) > max_x else (u_x+padding)
        cube = pred[l_z:u_z, l_y:u_y, l_x:u_x]
        print('Cube shape: ', cube.shape)
        io.imsave(os.path.join(save_path, cube_name+'_preds.tif'), cube, plugin='tifffile', check_contrast=False)

def cube_for_hyperparams_search_extraction(im_path, raw_pred_path, gt_path, cube_dict, padding, save_dir):
    print("cubes to be extracted ", cube_dict)
    # Image
    print('Extracting image cubes...')
    im = reading_data(im_path)
    im_save_path = os.path.join(save_dir, 'search_images')
    if not os.path.exists(im_save_path):
        os.makedirs(im_save_path)
    extract_cubes_for_hyperparams_search(im, cube_dict, im_save_path, padding)
    # Prediction
    print('Extracting prediction cubes...')
    pred = reading_data(raw_pred_path)
    pred_save_path = os.path.join(save_dir, 'search_preds')
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    extract_preds_for_hyperparams_search(pred, cube_dict, pred_save_path, padding)
    # gt 
    print('Moving ground truth cubes...')
    gt_save_path = os.path.join(save_dir, 'search_gt')
    if not os.path.exists(gt_save_path):
        os.makedirs(gt_save_path)
    for cube_name in cube_dict:
        if 'empty' in cube_name:
            continue
        cube_num = cube_name.split('_')[-1]
        gt_name = 'complete_label_' + cube_num + '.tif'
        gt_path = os.path.join(gt_path, gt_name)
        shutil.copy(gt_path, os.path.join(gt_save_path, cube_name+'_gt'+'.tif'))
    
    return im_save_path, pred_save_path, gt_save_path

def reading_stack_from_dir(im_stack_dir):
    file_type = im_stack_dir.split('/')[-1]
    files = natsorted(glob.glob(os.path.join(im_stack_dir, '*.'+file_type)))
    if file_type == 'tif':
        print('Reading tif files...')
        im_stack = []
        for d in tqdm.tqdm(files):
            im_stack.append(io.imread(d, plugin='tifffile'))
        im_stack = np.stack(im_stack, axis=0)

    elif file_type == 'jp2':
        print('Reading jp2 files...')
        im_stack = []
        for d in tqdm.tqdm(files):
            im_stack.append(io.imread(d))
        im_stack = np.stack(im_stack, axis=0)
    else:
        raise ValueError(f'The file type:{file_type} is not supported')
    return im_stack

def reading_stack_from_file(im_stack_path):
    file_type = im_stack_path.split('.')[-1]
    if file_type == 'tif':
        print('Reading tif files...')
        im_stack = io.imread(im_stack_path, plugin='tifffile')
    elif file_type == 'jp2':
        print('Reading jp2 files...')
        im_stack = io.imread(im_stack_path)
    else:
        raise ValueError(f'The file type:{file_type} is not supported')
    return im_stack

def reading_json(cube_coord_json_path, sample_name, extract_voi):
    with open(cube_coord_json_path, 'r') as f:
        json_dict = json.load(f)
    cube_dict = json_dict[sample_name][extract_voi]
    return cube_dict

def crop_roi(cube, cube_name, cube_dict, padding=32):
    '''
    This function crops the cube to the original size without padding.
    Args:
        cube: np.array, the input cube with padding
        cube_name: str, the name of the cube
        cube_dict: dict, the dictionary containing the coordinates of the cubes
        padding: int, the padding size
    Returns:
        cropped_cube: np.array, the cropped cube without padding
    '''
    if cube_name not in cube_dict.keys():
        raise ValueError(f'Cube name {cube_name} not found in cube_dict. Please check the name.')
    x_start, y_start, z_start = padding, padding ,padding 
    if cube_dict[cube_name]['x_range'][0] - padding < 0:
        x_start = cube_dict[cube_name]['x_range'][0]
    if cube_dict[cube_name]['y_range'][0] - padding < 0:
        y_start = cube_dict[cube_name]['y_range'][0]
    if cube_dict[cube_name]['z_range'][0] - padding < 0:
        z_start = cube_dict[cube_name]['z_range'][0]
    x_end, y_end, z_end = (x_start + cube_dict[cube_name]['x_range'][1] - cube_dict[cube_name]['x_range'][0],
                            y_start + cube_dict[cube_name]['y_range'][1] - cube_dict[cube_name]['y_range'][0],
                            z_start + cube_dict[cube_name]['z_range'][1] - cube_dict[cube_name]['z_range'][0])
    cropped_cube = cube[z_start:z_end, y_start:y_end, x_start:x_end]
    return cropped_cube

def extract_2_58_cube_hyperparams_search(cube_name, cube_info, img, save_path, padding=32):
    img = img
    padding = padding
    z, x, y = cube_info['coordinates']
    l_z = 0 if (z-padding) < 0 else (z-padding)
    l_y = 0 if (y-padding) < 0 else (y-padding)
    l_x = 0 if (x-padding) < 0 else (x-padding)
    u_z = 5504 if (z+512+padding) > 5504 else (z+512+padding)
    u_y = 1924 if (y+512+padding) > 1924 else (y+512+padding)
    u_x = 1924 if (x+512+padding) > 1924 else (x+512+padding)
    cube = img[l_z:u_z, l_y:u_y, l_x:u_x]
    print('Cube shape: ', cube.shape)
    io.imsave(os.path.join(save_path, cube_name, '_padding_', str(padding), '.tif'), cube, plugin='tifffile', check_contrast=False)

def extract_12_1_roi_hyperparams_search(cube_name, cube_info, img, save_path, extract_type='img'):
    img = img
    padding = cube_info['padding']
    l_z, u_z = (cube_info['crop_z_range'][0]+cube_info['z_range'][0], cube_info['crop_z_range'][0]+cube_info['z_range'][1])
    l_y, u_y = cube_info['y_range']
    l_x, u_x = cube_info['x_range']

    l_z = 0 if (l_z-padding) < 0 else (l_z-padding)
    l_y = 0 if (l_y-padding) < 0 else (l_y-padding)
    l_x = 0 if (l_x-padding) < 0 else (l_x-padding) 
    u_z = 3769 if (u_z+padding) > 3769 else (u_z+padding)
    u_y = 1898 if (u_y+padding) > 1898 else (u_y+padding)
    u_x = 1898 if (u_x+padding) > 1898 else (u_x+padding)
    cube = img[l_z:u_z, l_y:u_y, l_x:u_x]
    print('Cube shape: ', cube.shape)
    if cube_info['mask'] and extract_type == 'img':
        mask = np.zeros_like(cube)
        print('Mask shape: ', mask.shape)
        io.imsave(os.path.join(save_path, cube_name+'_gt'+'_padding_'+str(padding)+'.tif'), mask, plugin='tifffile', check_contrast=False)
    io.imsave(os.path.join(save_path, cube_name+'_padding_'+str(padding)+'.tif'), cube, plugin='tifffile', check_contrast=False)

def extract_25_08_roi_hyperparams_search(cube_name, cube_info, img, save_path, extract_type='img', padding=0):
    img = img
    padding = padding
    if extract_type == 'img':
        l_z, u_z = (cube_info['crop_z_range'][0]+cube_info['z_range'][0], cube_info['crop_z_range'][0]+cube_info['z_range'][1])
    elif extract_type == 'lbl':
        lbl_z_range = [765, 2762]
        l_z, u_z = (cube_info['crop_z_range'][0]-lbl_z_range[0]+cube_info['z_range'][0], cube_info['crop_z_range'][0]-lbl_z_range[0]+cube_info['z_range'][1])
    else:
        raise ValueError('The extract type is not supported')
    l_y, u_y = cube_info['y_range']
    l_x, u_x = cube_info['x_range']

    l_z = 0 if (l_z-padding) < 0 else (l_z-padding)
    l_y = 0 if (l_y-padding) < 0 else (l_y-padding)
    l_x = 0 if (l_x-padding) < 0 else (l_x-padding) 
    u_z = 2829 if (u_z+padding) > 2829 else (u_z+padding)
    u_y = 3412 if (u_y+padding) > 3412 else (u_y+padding)
    u_x = 3020 if (u_x+padding) > 3020 else (u_x+padding)
    cube = img[l_z:u_z, l_y:u_y, l_x:u_x]
    print('Cube shape: ', cube.shape)
    if 'empty' in cube_name:
        mask = np.zeros_like(cube)
        print('Mask shape: ', mask.shape)
        io.imsave(os.path.join(save_path, cube_name+'_gt'+'_padding_'+str(padding)+'.tif'), mask, plugin='tifffile', check_contrast=False)
    io.imsave(os.path.join(save_path, cube_name+'_padding_'+str(padding)+'.tif'), cube, plugin='tifffile', check_contrast=False)

