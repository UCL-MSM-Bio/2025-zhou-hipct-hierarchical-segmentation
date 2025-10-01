import os
from glob import glob
import json
import numpy as np
import skimage.io as skio
from natsort import natsorted
from metrics import generate_dice_scores, instance_dice
from skimage.morphology import  label
from napari_simpleitk_image_processing import label_statistics
from helper import generate_lhc_sample, bounds_calculation, get_minimum_vol, check_size_filtered_table, remove_labels, label_closing
from cube_extraction import cube_for_hyperparams_search_extraction, reading_json, crop_roi

PROPERTY_SELECTION = {
        '2.58':{
            'property': ['variance_low', 'variance_high', 'roundness'],
            'threshold': [(0, 0.3), (0.7, 1), (0.01, 0.99)]
            }
    }


def search_pipeline(im_path, raw_pred_path, gt_path, resolution, cube_coord_json_path, sample_name, voi, lhc_seed=0, lhc_sample_size=20, padding=0, save_dir=None):
    
    # first check the smallest glomeruli size filtered table. not exist, create one
    pred_props_table = check_size_filtered_table(im_path, raw_pred_path, resolution)
    
    extraction_save_dir = os.path.join(save_dir, (sample_name+'_'+voi), 'seed_'+str(lhc_seed)+'_padding_'+str(padding))

    # extract the evaluation cubes
    cube_dict = reading_json(cube_coord_json_path, sample_name, voi) 
    im_save_path, pred_save_path, gt_save_path = cube_for_hyperparams_search_extraction(im_path, raw_pred_path, gt_path, cube_dict, padding, extraction_save_dir)

    #pred_props_table = pd.read_csv(os.path.join(save_dir,'props_table_itk_after_size.csv'))
    print('Calculate the bounds...')
    # calculate the bounds
    bounds = bounds_calculation(pred_props_table, PROPERTY_SELECTION[str(resolution)])

    # sample the hyper-parameters according to the bounds
    print('Sampling the hyper-parameters...')
    sample_matrix = generate_lhc_sample(bounds, PROPERTY_SELECTION[str(resolution)], extraction_save_dir, lhc_seed, lhc_sample_size)
    print(sample_matrix)

    training_cube_path = im_save_path
    training_gt_path = gt_save_path
    training_pred_path = pred_save_path

    training_cubes = natsorted(glob(os.path.join(training_cube_path, '*.tif')))
    training_gt = natsorted(glob(os.path.join(training_gt_path, '*.tif')))
    training_preds = natsorted(glob(os.path.join(training_pred_path, '*.tif')))
    print(f'\nNumber of training cubes: {len(training_cubes)}\n')

    sample_keys = []
    json_dict = {}
    for cube_path, gt_path, pred_path in zip(training_cubes, training_gt, training_preds): 
        print(f'\nProcessing the cube: {cube_path}')
        print(f'GT path: {gt_path}')
        print(f'Prediction path: {pred_path}')

        cube_name = os.path.basename(cube_path).split('.')[0]
        print(f'Cube name: {cube_name}')
        sample_keys.append(cube_name)
        cube = skio.imread(cube_path, plugin='tifffile') 
        gt = skio.imread(gt_path, plugin='tifffile') 
        pred = skio.imread(pred_path, plugin='tifffile') 
        pred = label_closing(pred, cube_size=5)
        pred_lbl = label(pred)
        pred_props_table = label_statistics(cube, pred_lbl, shape=True, perimeter=True, position=True, moments=False)
        print(f'There are {len(pred_props_table)} objects detected in the prediction')
        
        gt_lbl = label(gt)
        cube_512 = crop_roi(cube, cube_name, cube_dict, padding)
        gt_props_table = label_statistics(cube_512, gt_lbl, shape=True, perimeter=True, position=True, moments=False)

        pred_512 = crop_roi(pred, cube_name, cube_dict, padding)

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
        for var_low, var_high,  roundness in sample_matrix:
            print(f' - Processing the hyper-parameters: var_low: {var_low}, var_high: {var_high}, roundness: {roundness}')
            if f'search_{search_idx}' not in json_dict.keys():
                json_dict[f'search_{search_idx}'] = {'var_low': var_low, 'var_high': var_high, 'roundness': roundness}

            pred_remove = []
            monitor = {'var_low': 0, 'var_high': 0, 'roundness': 0}
            for index, row in pred_props_table.iterrows():
                if row['roundness'] < roundness and row['number_of_pixels_on_border'] == 0:
                    pred_remove.append(int(row['label']))
                    monitor['roundness'] += 1
                if row['variance'] < var_low:
                    pred_remove.append(int(row['label']))
                    monitor['var_low'] += 1
                if row['variance'] > var_high:
                    pred_remove.append(int(row['label']))
                    monitor['var_high'] += 1

            print(f'Number of labels to be removed: {len(pred_remove)}')
            print(f'Number of labels removed based on roundness: {monitor["roundness"]}, var_low: {monitor["var_low"]}, var_high: {monitor["var_high"]}')
            
            pred_lbl_copy = pred_lbl.copy()
            filtered_pred = remove_labels(pred_remove, pred_props_table, pred_lbl_copy)
            filtered_pred[filtered_pred > 0] = 1
            filtered_pred = filtered_pred.astype(np.int8)
            filtered_pred_roi = crop_roi(filtered_pred, cube_name, cube_dict, padding)
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
                                                            'removed_based_on_var_low': monitor['var_low'],
                                                            'removed_based_on_var_high': monitor['var_high'],
                                                            'dice_before_processing': dice_before_processing,
                                                            'instance_dice_before_processing': instance_dice_before_processing,
                                                            'dice': dice_score,
                                                            'instance_dice_score': instance_dice_score,
                                                            'instance_dice_list': dice_list}
            search_idx += 1

    for search_id in json_dict.keys():
        mean_dice_before_processing = np.mean([json_dict[search_id][cube]['dice_before_processing'] for cube in sample_keys])
        mean_instance_dice_before_processing = np.mean([json_dict[search_id][cube]['instance_dice_before_processing'] for cube in sample_keys])
        mean_dice = np.mean([json_dict[search_id][cube]['dice'] for cube in sample_keys])
        mean_instance_dice = np.mean([json_dict[search_id][cube]['instance_dice_score'] for cube in sample_keys])
        json_dict[search_id]['mean_dice_before_processing'] = mean_dice_before_processing
        json_dict[search_id]['mean_instance_dice_before_processing'] = mean_instance_dice_before_processing
        json_dict[search_id]['mean_dice'] = mean_dice
        json_dict[search_id]['mean_instance_dice'] = mean_instance_dice
    json.dump(json_dict, open(os.path.join(extraction_save_dir,'search_result.json'), 'w'), indent=4)


if __name__ == '__main__':
    resolution = 2.58
    size_filtered = get_minimum_vol(resolution)

    print(f'\nWorking on resolution: {resolution} um, Size filter: {size_filtered} pixels, Property selection: {PROPERTY_SELECTION[str(resolution)]}\n')
    search_pipeline(
        im_path='/hdd/yang/data/kidney_seg/LADAF-2020-27_left/2.58um/original/tif_5504',
        raw_pred_path='/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/inference_whole_vol/pred_after_size_filtered.tif',
        gt_path='/hdd2/yang/projects/glomeruli_segmentation/2025-zhou-hipct-hierarchical-segmentation/data/manual_annotations/highres_training_16bit_labels/',
        resolution=resolution, 
        cube_coord_json_path='/hdd2/yang/projects/glomeruli_segmentation/2025-zhou-hipct-hierarchical-segmentation/segmentation/postprocessing/parameter_search/cube_coordinates.json', 
        sample_name='LADAF-2020-27_Left', 
        voi='cube_2-58', 
        lhc_seed=0, 
        lhc_sample_size=20,
        padding=32, 
        save_dir='/hdd2/yang/projects/glomeruli_segmentation/2025-zhou-hipct-hierarchical-segmentation/data/parameter_search/'
        )