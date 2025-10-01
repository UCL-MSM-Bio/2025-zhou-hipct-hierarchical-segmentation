import skimage.io as skio
import numpy as np
from natsort import natsorted
from skimage.morphology import label
from napari_simpleitk_image_processing import label_statistics
import time
from tqdm import tqdm
import os
import glob
import parameter_search.helper as helper
import pandas as pd

# Setting up the parameters from parameter search
VAR_LOW = 114770.71311622646
VAR_HIGH = 3069051.732661265
ROUNDNESS = 0.7091737266764246
SIZE_IN_PIXEL = helper.get_minimum_vol(resolution=2.58)

# function should be tailored according to the parameter search results
def remove(instance):
    flag = False
    if instance['variance'] < VAR_LOW or instance['variance'] > VAR_HIGH:
        flag = True
    elif instance['number_of_pixels'] < SIZE_IN_PIXEL and instance['number_of_pixels_on_border'] == 0:
        flag = True
    elif instance['roundness'] < ROUNDNESS and instance['number_of_pixels_on_border'] == 0:
        flag = True
    return flag

def pipeline_2_58(data_dir, pred_dir, output_dir, label_csv_file=None):
    '''
    Post processing pipeline for 2.58um data
    Args:
        data_path: path to the data
        label_path: path to the label
        output_path: path to save the output
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    print('Loading data...')
    t_s = time.time()
    if data_dir.endswith('.tif'):
        data = skio.imread(data_dir, plugin='tifffile')
    else:
        print('Data is a folder, loading all the slices...')
        data_slices = natsorted(glob.glob(os.path.join(data_dir, '*.tif')))
        data = np.stack([skio.imread(d, plugin='tifffile') for d in data_slices], axis=0)

    pred = skio.imread(os.path.join(pred_dir, 'clahe_vol_000.tif'), plugin='tifffile')
    data = data[:-1, :, :] # remove the last slice as the normalisation issue
    pred = pred[:-1, :, :] # remove the last slice as the normalisation issue
    print(f'Data shape: {data.shape}')
    print(f'Prediction shape: {pred.shape}')
    t_e = time.time()
    print(f'Time to load data: {t_e - t_s} seconds \n')

    print('generate label files...')
    t_s = time.time()
    pred_lbl = label(pred)

    # generate label files
    if label_csv_file is None:
        props_table_itk = label_statistics(data, pred_lbl, shape=True, perimeter=True, position=True, moments=False)
        print(f'There are {len(props_table_itk)} objects detected in the label')
        t_e = time.time()
        print(f'Time to generate label files: {t_e - t_s} seconds \n')
        # save the props_table_itk
        props_table_itk.to_csv(os.path.join(output_dir, 'props_table_itk_before.csv'))

    else:
        props_table_itk = pd.read_csv(label_csv_file)

    # Obtained the outliers need to be removed
    label_removed = []
    print('Start removing outliers...')
    for index, row in props_table_itk.iterrows():
        if remove(row):
            label_removed.append(int(row['label']))
    print('Removing ratio: \n', len(label_removed)/len(props_table_itk))

    # Remove the outliers
    print('Removing labels...')
    pbar= tqdm(total=len(label_removed))
    for l in label_removed:
        pbar.set_description(f'Processing:')
        row = props_table_itk[props_table_itk['label']==l]
        x_start = row['bbox_0'].values[0]
        y_start = row['bbox_1'].values[0]
        z_start = row['bbox_2'].values[0]
        x_range = row['bbox_3'].values[0]
        y_range = row['bbox_4'].values[0]
        z_range = row['bbox_5'].values[0]
        extracted_cube = pred_lbl[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range].copy()
        extracted_cube[extracted_cube == l] = 0
        pred_lbl[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range] = extracted_cube
        pbar.update(1)
    del pbar
    pred_lbl[pred_lbl > 0] = 1
    lbl_processed = pred_lbl.astype(np.int8)

    # closing operation
    print('Performing closing operation...')
    lbl_processed = helper.label_closing(lbl_processed, cube_size=3)
    
    # Save the processed label
    print('Saving the processed label...')
    skio.imsave(os.path.join(output_dir, 'label_after_post-processed.tif') , lbl_processed, plugin='tifffile', check_contrast=False)
    print(f'Processed label saved at {output_dir}')


if __name__ == '__main__':
    print('Post-processing for 2.58um data...')
    print(f'Parameters: low variance {VAR_LOW}, high variance {VAR_HIGH}, roundness {ROUNDNESS}, size in pixel {SIZE_IN_PIXEL}')

    data_dir = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/2.58um/original/tif'
    pred_dir = '/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/inference_whole_vol'
    output_dir = '/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/inference_whole_vol/post_processed/'
    label_csv_file = None
    pipeline_2_58(data_dir, pred_dir, output_dir, label_csv_file)