import numpy as np
import skimage.io as skio
import glob
import os
from natsort import natsorted
import skimage.io as skio
import json
import shutil
from tqdm import tqdm

def seed_everything(seed):
    np.random.seed(seed)

def all_zero(lb_file):
    '''
    Check if the label is all zero
    Args:
        lb_file: label file, matrix
    '''
    if np.sum(lb_file) == 0:
        return True
    else:
        return False

def load_label(path):
    '''
    Load .tif file which is generated from Amira with skimage.io
    Args:
        path: path of .tif file
    '''
    tif_stack = skio.imread(path, plugin='tifffile')
    return tif_stack

def selecting_cube(lb, threshold=0):
    '''
    selecting the cube with label percentage >= threshold
    Args:
        lb_file: label file, matrix
    '''
    total_pixels = lb.shape[0] * lb.shape[1] * lb.shape[2]
    num_non_zero = np.sum(lb)
    label_percentage = num_non_zero / total_pixels
    if label_percentage > threshold:
        return True 
    else:
        return False

def generate_training_json_file(lb_path, output_path, percetage_threshold=0, keep_prob=0):
    '''
    Save the samples with valid label into json file
    Args:
        percetage_threshold: The minimum label percentage to keep
        keep_prob: The probability of keeping the samples with threshold percentage
    '''
    p = 0
    output_dict = {'training': [], 'percentage_threshold': percetage_threshold, 'keep_prob': keep_prob}
    lbl_files = natsorted(glob.glob(os.path.join(lb_path, '*.tif')))
    print('Number of label files:', len(lbl_files))
    for i, lb_file in enumerate(tqdm(lbl_files)):
        lb = load_label(lb_file)
        lb_name = os.path.basename(lb_file)

        selected_flag = selecting_cube(lb, percetage_threshold)
        if not selected_flag and np.random.rand() < keep_prob:
            selected_flag = True
            p += 1

        if selected_flag:
            output_dict['training'].append(lb_name.replace('label', 'cube'))

    output_dict['keep_prob'] = p / len(lbl_files)
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=4)
    print('Number of training samples:', len(output_dict['training']))

def gen_JSON(dataTr, dataset_name, save_path):
    data = {}
    data['name'] = dataset_name
    data['description'] = 'Glomerulus segmentation'
    data['Reference'] = 'UCL MSMaH Bio'
    data['licence'] = 'CC-BY-SA 4.0'
    data['release'] = '0.0.1'
    data['tensorImageSize'] = '3D'
    data['channel_names'] = {
        '0': 'HiPCT'
    }
    data['labels'] = {
        'background': 0,
        'glomerulus': 1
    }
    data['numTraining'] = len(dataTr)
    data['file_ending'] = '.tif'
    data["overwrite_image_reader_writer"] = "Tiff3DIO"
    
    with open(os.path.join(save_path, 'dataset.json'), 'w') as f:
        json.dump(data, f, indent=4)

    return data

def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def processing_dataset(raw_data_path, nnunet_data_path, dataset_name, raw_data_vilid_json, train_des_folder, train_label_des_folder):
    
    if not os.path.exists(train_des_folder):
        os.makedirs(train_des_folder)
    if not os.path.exists(train_label_des_folder):
        os.makedirs(train_label_des_folder)
    
    with open(raw_data_vilid_json, 'r') as f:
        raw_data_seperate = json.load(f)
    train_data = []
    train_labels = []
    for item in raw_data_seperate['training']:
        train_data.append(item)
        train_labels.append(item.replace('cube', 'label'))
    natsorted(train_data)
    natsorted(train_labels)
    print('Number of training samples:', len(train_data))
    print('Number of training labels:', len(train_labels))
    spacing = (1, 1, 1)
    pbar = tqdm(total=len(train_data))
    for i, (cube_name, lbl_name) in enumerate(zip(train_data, train_labels)):
        pbar.update(1)
        target_name = cube_name.split('.')[0] + f'_{i:03d}' 
        target_lbl_name = lbl_name.split('.')[0] + f'_{i:03d}'
        original_cube_path = os.path.join(raw_data_path, 'cubes', cube_name)
        shutil.copy(original_cube_path, os.path.join(train_des_folder, target_name + '_0000.tif'))
        save_json({'spacing': spacing}, os.path.join(train_des_folder, target_name + '.json'))

        original_label_path = os.path.join(raw_data_path, 'labels', lbl_name)
        shutil.copy(original_label_path, os.path.join(train_label_des_folder, target_lbl_name.replace('label', 'cube') + '.tif'))
        # spacing file!
        save_json({'spacing': spacing}, os.path.join(train_label_des_folder, target_lbl_name.replace('label', 'cube') + '.json'))
    pbar.close()

    # now we generate the dataset json
    gen_JSON(
        dataTr=train_data,
        dataset_name=dataset_name,
        save_path=os.path.join(nnunet_data_path, 'nnUNet_raw', train_des_folder.split('/')[-2])
    )

def generate_training_data(im_path, lbl_path, crop_save_path, overlap=0.5, patch_size=64):
    """
    Generate training data by cropping the volume into patches
    Args:
        im_path: low resolution image, 8bit, tif
        lbl_path: registered image, 8bit, tif
        crop_save_path: save path for cropped patches
        overlap: overlap ratio between patches
        patch_size: size of the cubic patch
    """
    if not os.path.exists(os.path.join(crop_save_path, 'cubes')):
        os.makedirs(os.path.join(crop_save_path, 'cubes'))
    if not os.path.exists(os.path.join(crop_save_path, 'labels')):
        os.makedirs(os.path.join(crop_save_path, 'labels'))
    im = skio.imread(im_path, plugin='tifffile')
    lbl = skio.imread(lbl_path, plugin='tifffile')
    assert im.shape == lbl.shape, 'Image and label shape mismatch'
    z, y, x = im.shape
    print('Image shape:', im.shape)
    stride = int(patch_size * overlap)
    z_pad = stride - (z % stride)
    y_pad = stride - (y % stride)
    x_pad = stride - (x % stride)
    paded_im = np.pad(im, ((z_pad//2, z_pad - z_pad//2), (y_pad//2, y_pad - y_pad//2), (x_pad//2, x_pad - x_pad//2)), 'constant')
    paded_lbl = np.pad(lbl, ((z_pad//2, z_pad - z_pad//2), (y_pad//2, y_pad - y_pad//2), (x_pad//2, x_pad - x_pad//2)), 'constant')
    z, y, x = paded_im.shape
    print('Padded image shape:', paded_im.shape)
    print('Padded label shape:', paded_lbl.shape)
    print('Padded value:', paded_im[0,0,0])
    print('Generating patches...')
    for i in range(0, z, stride):
        for j in range(0, y, stride):
            for k in range(0, x, stride):
                patch_im = paded_im[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                patch_lbl = paded_lbl[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                patch_name = f'cube_z{i}_y{j}_x{k}.tif'
                label_name = f'label_z{i}_y{j}_x{k}.tif'
                skio.imsave(os.path.join(crop_save_path, 'cubes', patch_name), patch_im, plugin='tifffile', check_contrast=False)
                skio.imsave(os.path.join(crop_save_path, 'labels', label_name), patch_lbl, plugin='tifffile', check_contrast=False)
        


if __name__ == '__main__':
    seed_everything(42)
    base = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/12.1um'
    # 1. Crop data
    im_path = os.path.join(base, 'training/training_vol_roi/low_res_vol_roi.tif')
    lbl_path = os.path.join(base, 'training/training_vol_roi/registered_lbl_roi.tif')
    crop_save_path = os.path.join(base, 'training/')
    generate_training_data(im_path, lbl_path, crop_save_path, overlap=0.5, patch_size=128)
    
    # 2. Generate json file for training cubes based on a propability 
    lb_path = os.path.join(crop_save_path, 'labels')
    if not os.path.exists(lb_path):
        os.makedirs(lb_path)
    p_threshold = 0 # percentage threshold: keep cubes with label percentage > p_threshold
    prob = 0 # keeping probability for cubes with label percentage < p_threshold
    output_path = os.path.join(base, f'training/valid_threshold_{str(p_threshold)}_prob_{str(prob)}.json')
    generate_training_json_file(lb_path, output_path,  percetage_threshold=p_threshold, keep_prob=prob)
    prob = 0.05
    output_path = os.path.join(base, f'training/valid_threshold_{str(p_threshold)}_prob_{str(prob)}.json')
    generate_training_json_file(lb_path, output_path,  percetage_threshold=p_threshold, keep_prob=prob)

    # 3. Move Images
    raw_data_path = crop_save_path
    nnunet_data_path = '/hdd/yang/data/kidney_seg_nnunet'
    json_file = os.path.join(raw_data_path, 'valid_threshold_0_prob_0.05.json') # use the json file generated above to generate training dataset
    train_des_folder = os.path.join(nnunet_data_path, 'nnUNet_raw/Dataset005_12-1Glom_search_w_fat/imageTr')
    train_label_des_folder = os.path.join(nnunet_data_path, 'nnUNet_raw/Dataset005_12-1Glom_search_w_fat/labelsTr')
    dataset_name = 'Glomerulus-12.1um'
    processing_dataset(raw_data_path, nnunet_data_path, dataset_name, json_file, train_des_folder, train_label_des_folder)
    
