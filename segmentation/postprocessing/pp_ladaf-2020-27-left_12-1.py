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
from postprocessor import remove_on_density


# Setting up the parameters from parameter search
VAR_LOW  =  179663.37055017683
ROUNDNESS = 0.6823638873197536
SIZE_IN_PIXEL = helper.get_minimum_vol(resolution=12.1)
WINDOW_SIZE = 128
DENSITY_THRESHOLD = 3

# function should be tailored according to the parameter search results
def remove(instance, pred_lbl):
    flag = False
    if instance['number_of_pixels'] < SIZE_IN_PIXEL and instance['number_of_pixels_on_border'] == 0:
        flag = True
    if instance['variance'] < VAR_LOW:
        flag = True
    if instance['roundness'] < ROUNDNESS and instance['number_of_pixels_on_border'] == 0:
        if instance['number_of_pixels'] > 1.5 * SIZE_IN_PIXEL:
            x_start = int(instance['bbox_0'])
            y_start = int(instance['bbox_1'])
            z_start = int(instance['bbox_2'])
            x_range = int(instance['bbox_3'])
            y_range = int(instance['bbox_4'])
            z_range = int(instance['bbox_5'])
            extracted_cube = pred_lbl[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range].copy()
            extracted_cube[extracted_cube != instance['label']] = 0
            extracted_cube[extracted_cube > 0] = 1
            extracted_cube = extracted_cube.astype(np.uint8)
            if helper.watershed_big_label(extracted_cube, SIZE_IN_PIXEL) < 2:
                flag = True
        else:
            flag = True
    return flag

        

def pipeline_12_1(data_dir, pred_dir, output_dir, label_csv_file=None): #remove_based_on_density=True, density_threshold=5):
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
    print('\nStart removing outliers...')
    for index, row in props_table_itk.iterrows():
        if remove(row, pred_lbl):
             label_removed.append(int(row['label']))
    print('\n removing ratio: ', len(label_removed)/len(props_table_itk))
    
    # remove the outliers
    print('\nRemoving labels...')
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
    
    # removed based on the glomeruli density
    print(f'\nStart removing outliers based on density, threshold: {DENSITY_THRESHOLD} ...')
    label_removed_density = []
    pred_lbl = label(lbl_processed)
    props_table_itk = label_statistics(data, pred_lbl, shape=False, perimeter=False, position=True, moments=False)
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

    # Save the processed label
    print('Saving the processed label...')
    skio.imsave(os.path.join(output_dir, 'label_after_post-processed.tif'), lbl_processed, plugin='tifffile', check_contrast=False)
    print(f'Processed label saved at {output_dir}')




if __name__ == '__main__':

    print('Post-processing for 12.1um data...')
    print(f'Parameters: low variance {VAR_LOW}, roundness {ROUNDNESS}, size in pixel {SIZE_IN_PIXEL}, window size {WINDOW_SIZE}, density threshold {DENSITY_THRESHOLD}')

    data_dir = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/12.1um/original/tif'
    pred_dir = '/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol'
    output_dir = '/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol/post_processed/'
    label_csv_file = None #os.path.join(output_dir, 'props_table_itk_before.csv')
    pipeline_12_1(data_dir, pred_dir, output_dir, label_csv_file)
    

