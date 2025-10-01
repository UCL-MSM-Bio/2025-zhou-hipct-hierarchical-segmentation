import os
import skimage.io as skio
from skimage.morphology import label
from napari_simpleitk_image_processing import label_statistics
import pandas as pd
import time
import numpy as np 
from tqdm import tqdm
from natsort import natsorted
import glob
import parameter_search.helper as helper
from postprocessor import remove_on_density, apply_mask

# Setting up the parameters
WINDOW_SIZE = 64
DENSITY_THRESHOLD = 3
SIZE_IN_PIXEL = helper.get_minimum_vol(resolution=25.08)

        

def pipeline_25_08(data_dir, pred_dir, mask_path, output_dir, label_csv_file=None): #remove_based_on_density=True, density_threshold=5):
    '''
    Post processing pipeline for 12.1um data
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
        data_slices = data_slices[765:2763] # This is the range for LADAF-2020-27-left 25.08um data
        data = np.stack([skio.imread(d, plugin='tifffile') for d in data_slices], axis=0)

    pred = skio.imread(os.path.join(pred_dir, 'pred_after_size_filtered_masked.tif'), plugin='tifffile')
    print(f'Data shape: {data.shape}')
    print(f'Prediction shape: {pred.shape}')
    t_e = time.time()
    print(f'Time to load data: {t_e - t_s} seconds \n')

    print('generate label files...')
    t_s = time.time()
    pred_lbl = label(pred)

    # generate label files
    if label_csv_file is None:
        props_table_itk = label_statistics(data, pred_lbl, shape=False, perimeter=False, position=True, moments=False)
        print(f'There are {len(props_table_itk)} objects detected in the label')
        t_e = time.time()
        print(f'Time to generate label files: {t_e - t_s} seconds \n')
        # save the props_table_itk
        props_table_itk.to_csv(os.path.join(output_dir, 'pred_after_size_filtered_masked.csv'))
    else:
        props_table_itk = pd.read_csv(label_csv_file)

    
    # removed based on the glomeruli density
    print(f'\nStart removing outliers based on density, threshold: {DENSITY_THRESHOLD} ...')
    label_removed_density = []
    print(f'There are {len(props_table_itk)} objects detected in the label')
    pbar = tqdm(total=len(props_table_itk))
    for index, row in props_table_itk.iterrows():
        if remove_on_density(row, 
                        props_table_itk['centroid_0'], 
                        props_table_itk['centroid_1'], 
                        props_table_itk['centroid_2'], 
                        shape=data.shape, size=WINDOW_SIZE, threshold=DENSITY_THRESHOLD):
            label_removed_density.append(int(row['label']))
        pbar.update(1)
    del pbar
    print('\n removing ratio: ', len(label_removed_density)/len(props_table_itk))
    
    print('\nRemoving labels...')
    pbar= tqdm(total=len(label_removed_density))
    for l in  label_removed_density:
        pbar.set_description(f'Processing:')
        lbl_row = props_table_itk[props_table_itk['label']==l]
        x_start = lbl_row['bbox_0'].values[0]
        y_start = lbl_row['bbox_1'].values[0]
        z_start = lbl_row['bbox_2'].values[0]
        x_range = lbl_row['bbox_3'].values[0]
        y_range = lbl_row['bbox_4'].values[0]
        z_range = lbl_row['bbox_5'].values[0]
        extracted_cube = pred_lbl[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range].copy()
        extracted_cube[extracted_cube == l] = 0
        pred_lbl[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range] = extracted_cube
        pbar.update(1)
    del pbar
    pred_lbl[pred_lbl > 0] = 1
    lbl_processed = pred_lbl.astype(np.int8)
    del pred_lbl
    del props_table_itk
    
    # apply the mask
    print('\nApplying the mask...')
    pred_processed_masked = apply_mask(mask_path, lbl_processed)

    # Save the processed label
    print('Saving the processed label...')
    skio.imsave(os.path.join(output_dir, 'label_masked_after_post_processed.tif'), pred_processed_masked, plugin='tifffile', check_contrast=False)
    print(f'Processed label saved at {output_dir}')


if __name__ == '__main__':
    print('Post-processing for 25.08um data...')

    data_dir = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/original/tif'
    pred_dir = '/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset010_25-08Glom_search_w_fat_label_partly_filtered/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_4/inference_whole_vol'
    mask_path = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/mask/whole_kidney_mask_25.08um.tif'
    output_dir = '/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset010_25-08Glom_search_w_fat_label_partly_filtered/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_4/inference_whole_vol/post_processed'
    label_csv_file = None 
    pipeline_25_08(data_dir, pred_dir, mask_path, output_dir, label_csv_file)

