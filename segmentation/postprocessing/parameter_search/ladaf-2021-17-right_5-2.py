import os
import tqdm
from glob import glob
import json
import numpy as np
import pandas as pd
import skimage.io as skio
from natsort import natsorted
from metrics import generate_dice_scores, instance_dice
from skimage.morphology import closing, cube, label, remove_small_holes
from napari_simpleitk_image_processing import label_statistics
from helper import remove_based_on_size, label_closing, remove_labels, update_props_table, bounds_calculation, get_minimum_vol
from segmentation.models.nnUNet.nnunetv2.utilities import helpers

def search_pipeline(im_path, raw_pred_path, save_dir=None):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(os.path.join(os.path.dirname(raw_pred_path), 'props_table_itk_after_size.csv')):
        print('The properties table exists, using the existing one...')
        pred_props_table = pd.read_csv(os.path.join(os.path.dirname(raw_pred_path), 'props_table_itk_after_size.csv'))
    else:
        print('Reading data...')
        if im_path.endswith('.tif'):
            im = skio.imread(im_path, plugin='tifffile')
        else:
            print('Data is a folder, loading all the slices...')
            im_slices = natsorted(glob(os.path.join(im_path, '*.tif')))
            im = np.stack([skio.imread(slice, plugin='tifffile') for slice in im_slices], axis=0)

        im = im[:-1, :, :] # remove the last slice as the normalisation issue

        #read the raw prediction
        raw_pred = skio.imread(raw_pred_path, plugin='tifffile')
        raw_pred = raw_pred[:-1, :, :] # remove the last slice as the normalisation issue
        print(f'Image shape: {im.shape}')
        print(f'Raw prediction shape: {raw_pred.shape}\n')

        # generate label properties
        print('Generating prediction properties...')
        lbl_pred = label(raw_pred)
        pred_props_table = label_statistics(im, lbl_pred, shape=True, perimeter=True, position=True, moments=False)

        print(f'There are {len(pred_props_table)} objects detected in the label')
        # first remove based on size
        pred_removed = []
        print('Start removing outliers...')
        for index, row in pred_props_table.iterrows():
            if remove_based_on_size(row):
                pred_removed.append(int(row['label']))

        removed_based_on_size = len(pred_removed)
        print('Removing ratio: \n', removed_based_on_size/len(pred_props_table))

        pred_after_size_filtered = remove_labels(pred_removed, pred_props_table, lbl_pred)
        pred_after_size_filtered[pred_after_size_filtered > 0] = 1
        pred_after_size_filtered = pred_after_size_filtered.astype(np.int8)

        skio.imsave(os.path.join(save_dir, 'pred_after_size_filtered.tif'), pred_after_size_filtered, check_contrast=False)
        print('Finished the size filtered\n')
        del lbl_pred

        # update the properties table
        pred_props_table = update_props_table(pred_removed, pred_props_table)
        print(f'There are {len(pred_props_table)} objects detected in the label')
        # save the props_table_itk
        pred_props_table.to_csv(os.path.join(save_dir,'props_table_itk_after_size.csv'))

    #######################
    #######################

    print('Calculate the bounds...')
    # calculate the bounds
    bounds = bounds_calculation(pred_props_table, PROPERTY_SELECTION[str(RESOLUTION)])

    # sample the hyper-parameters according to the bounds
    print('Sampling the hyper-parameters...')
    l_bounds, u_bounds = [], []
    for prop in PROPERTY_SELECTION[str(RESOLUTION)]['property']: 
        l_bounds.append(bounds[prop][0])
        u_bounds.append(bounds[prop][1])
    sample_matrix = helpers.LHC(sample_size=20, dim=len(PROPERTY_SELECTION[str(RESOLUTION)]['property']), l_bounds=l_bounds, u_bounds=u_bounds)
    print('Sample_matrix shape: ', sample_matrix.shape)
    print('Finished the hyper-parameters sampling\n')
    with open(os.path.join(save_dir, 'sample_matrix.csv'), 'w') as f:
        for row in sample_matrix:
            f.write(','.join([str(r) for r in row]) + '\n')

    print(sample_matrix)
    json_dict = {}

    training_cube_path = '/hdd/yang/data/kidney_seg/LADAF-2021-17_right/5.2um_LADAF-2021-17_kidney-right_VOI-02.1/extraction/cubes'
    training_gt_path = '/hdd/yang/data/kidney_seg/LADAF-2021-17_right/5.2um_LADAF-2021-17_kidney-right_VOI-02.1/extraction/gt'
    training_pred_path = '/hdd/yang/data/kidney_seg/LADAF-2021-17_right/5.2um_LADAF-2021-17_kidney-right_VOI-02.1/extraction/preds'

    training_cubes = natsorted(glob(os.path.join(training_cube_path, '*.tif')))
    training_gt = natsorted(glob(os.path.join(training_gt_path, '*.tif')))
    training_preds = natsorted(glob(os.path.join(training_pred_path, '*.tif')))
    print(f'\nNumber of training cubes: {len(training_cubes)}\n')

    sample_keys = []
    for cube_path, gt_path, pred_path in zip(training_cubes, training_gt, training_preds): 
        print(f'\nProcessing the cube: {cube_path}')
        print(f'GT path: {gt_path}')
        print(f'Prediction path: {pred_path}')

        cube_name = os.path.basename(cube_path).split('.')[0]
        #print(f'Cube name: {cube_name}')
        sample_keys.append(cube_name)
        cube = skio.imread(cube_path, plugin='tifffile') 
        gt = skio.imread(gt_path, plugin='tifffile') 
        pred = skio.imread(pred_path, plugin='tifffile') 
        pred = label_closing(pred, cube_size=5)

        pred_lbl = label(pred)
        pred_props_table = label_statistics(cube, pred_lbl, shape=True, perimeter=True, position=True, moments=False)
        print(f'There are {len(pred_props_table)} objects detected in the prediction')
        
        gt_lbl = label(gt)
        cube_512 = crop_roi_5_2(cube, cube_name)
        gt_props_table = label_statistics(cube_512, gt_lbl, shape=True, perimeter=True, position=True, moments=False)

        pred_512 = crop_roi_5_2(pred, cube_name)

        if gt.sum() == 0:
            if pred_512.sum() == 0:
                dice_before_processing = 1
                instance_dice_before_processing = 1
            else:
                dice_before_processing = 0
                instance_dice_before_processing = 0
        else:
            dice_before_processing = generate_dice_scores(gt, pred_512)
            instance_dice_before_processing, _ = instance_dice(gt, pred_512, gt_props_table)

        del cube
        del pred
        del cube_512
        del gt_lbl

        search_idx = 1
        for roundness in sample_matrix:
            print(f' - Processing the hyper-parameters: roundness: {roundness}')
            if f'search_{search_idx}' not in json_dict.keys():
                json_dict[f'search_{search_idx}'] = {'roundness': roundness.tolist()}

            pred_remove = []
            #monitor = {'var_low': 0, 'var_high': 0, 'roundness': 0}
            monitor = {'roundness': 0}
            for index, row in pred_props_table.iterrows():
                if row['roundness'] < roundness and row['number_of_pixels_on_border'] == 0:
                    pred_remove.append(int(row['label']))
                    monitor['roundness'] += 1
                # if row['variance'] < var_low:
                #     pred_remove.append(int(row['label']))
                #     monitor['var_low'] += 1
                # if row['variance'] > var_high:
                #     pred_remove.append(int(row['label']))
                #     monitor['var_high'] += 1

            print(f'Number of labels to be removed: {len(pred_remove)}')
            print(f'Number of labels removed based on roundness: {monitor["roundness"]}')
            
            pred_lbl_copy = pred_lbl.copy()
            filtered_pred = remove_labels(pred_remove, pred_props_table, pred_lbl_copy)
            filtered_pred[filtered_pred > 0] = 1
            filtered_pred = filtered_pred.astype(np.int8)
            filtered_pred_roi = crop_roi_5_2(filtered_pred, cube_name)
            del pred_lbl_copy
            #del filtered_lbl    

            # iou and dice are not good metrics for empty label (no positive)
            if gt.sum() == 0:
                if filtered_pred_roi.sum() == 0:
                    instance_dice_score = 1
                    dice_score = 1
                else:
                    instance_dice_score = 0
                    dice_score = 0
                dice_list = []
            else:
                dice_score = generate_dice_scores(gt, filtered_pred_roi)
                instance_dice_score, dice_list = instance_dice(gt, filtered_pred_roi, gt_props_table)

            print(f'Dice score: {dice_score}')
            print(f'Instance dice score: {instance_dice_score}')

            json_dict[f'search_{search_idx}'][cube_name] = {'image_path': cube_path,
                                                            'gt_path': gt_path,
                                                            'pred_path': pred_path,
                                                            'removed_based_on_roundness': monitor['roundness'],
                                                            'dice_before_processing': dice_before_processing,
                                                            'instance_dice_before_processing': instance_dice_before_processing,
                                                            'dice': dice_score,
                                                            'instance_dice_score': instance_dice_score,
                                                            'instance_dice_list': dice_list}
            search_idx += 1

    #print(json_dict)
    for search_id in json_dict.keys():
        mean_dice_before_processing = np.mean([json_dict[search_id][cube]['dice_before_processing'] for cube in sample_keys])
        mean_instance_dice_before_processing = np.mean([json_dict[search_id][cube]['instance_dice_before_processing'] for cube in sample_keys])
        mean_dice = np.mean([json_dict[search_id][cube]['dice'] for cube in sample_keys])
        mean_instance_dice = np.mean([json_dict[search_id][cube]['instance_dice_score'] for cube in sample_keys])
        json_dict[search_id]['mean_dice_before_processing'] = mean_dice_before_processing
        json_dict[search_id]['mean_instance_dice_before_processing'] = mean_instance_dice_before_processing
        json_dict[search_id]['mean_dice'] = mean_dice
        json_dict[search_id]['mean_instance_dice'] = mean_instance_dice
    json.dump(json_dict, open(os.path.join(save_dir,'search_result.json'), 'w'), indent=4)



if __name__ == '__main__':

    RESOLUTION = 5.2
    PROPERTY_SELECTION = {
        '5.2':{
            'property': ['roundness'],
            'threshold': [(0.001, 0.05)]
            }
    }
    SIZE_FILTER = get_minimum_vol(resolution=RESOLUTION)

    print(f'\nWorking on resolution: {RESOLUTION} um, Size filter: {SIZE_FILTER} pixels, Property selection: {PROPERTY_SELECTION[str(RESOLUTION)]}\n')
    search_pipeline(
        im_path='/hdd/yang/data/kidney_seg/LADAF-2020-27_left/2.58um/original/tif',
        raw_pred_path='/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/inference_whole_vol/clahe_vol_000.tif',
        save_dir='/hdd/yang/results/glomeruli_segmentation/param_search/2.58um_seed0_computed_size'
        )