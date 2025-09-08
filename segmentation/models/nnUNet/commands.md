# Install
1. clone nnunet_repo
2. Change the path.py to customised directories
3. install apex
```console
cd nnUNet
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```
4. install nnunetv2
```
pip install -e .
```

# Train
1. Before training, extract dataset fingerprint(im size, voxel spacings, intensity information etc.) to build network structure
```Console
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
*DATASET_ID: dataset id without '00'*

Example:
```Console
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

 2. Train the model:
 ```Console
 nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
 ```
 *DATASET_NAME_OR_ID: dataset id without '00'* \
 *UNET_CONFIGURATION: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres*\
 *FOLD: 0, 1, 2, 3, 4 (default 5-fold)*
 
 Example:
 ```Console
 nnUNetv2_train 1 3d_fullres 0
 ```

# Inference
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f FOLD --save_probabilities
```

Example:
```Console
nnUNetv2_predict -i /hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset001_Glomeruli/imagesTs -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/inference_test -d 1 -c 3d_fullres -f 0
```
```Console
nnUNetv2_predict -i /hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset001_Glomeruli/imagesTs_whole -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/inference_sliding -d 1 -c 3d_fullres -f 1 -step_size 0.5
```

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/2.58um/full_8bits_tif/whole_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/inference_whole_vol -d 1 -c 3d_fullres -f 1 -step_size 0.5

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/12.1um/full_8bits_tif/whole_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset002_12-1Glom/nnUNetTrainer__nnUNetPlans12-1__3d_fullres/fold_0/inference_whole_vol -d 2 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans12-1

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/12.1um/full_8bits_tif/whole_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset003_12-1Glom_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol -d 3 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat 

# Postprocessing
```
nnUNetv2_apply_postprocessing -i E:\Kidnet_seg\prediction\nnUNet -o E:\Kidnet_seg\prediction\nnUNet_processed --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

### Examples:

```
nnUNetv2_predict -i /media/yang/LaCie/Data/Glomeruli\ Segmentation/correlative_dataset/2.58um_LADAF_2020-27_Left_Kidney/8bits_128_128_128_clahe -o /media/yang/LaCie/Glomeruli\ segmentation/models/nnUNet/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/predictions_on_2.58um_whole_data -d 1 -c 3d_fullres -f 0
```

```
nnUNetv2_predict -i /hdd/yang/data/kidney_seg/nnUNet_raw/Dataset001_Glomeruli/imagesTs_Jamie -o /hdd/yang/data/kidney_seg/nnUNet_predictions/prediction_on_test -d 1 -c 3d_fullres -f 0
```

```
nnUNetv2_predict -i /hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset001_Glomeruli/imagesTs_Jamie -o /hdd/yang/data/kidney_seg/nnUNet_predictions/prediction_on_jamie -d 1 -c 3d_fullres -f 0
```

nnUNetv2_move_plans_between_datasets -s 1 -t 2 -sp nnUNetPlans -tp nnUNetPlans12-1

nnUNetv2_move_plans_between_datasets -s 3 -t 1 -sp nnUNetPlans_w_fat -tp nnUNetPlans

nnUNetv2_preprocess -d 3 -plans_name uuUNetPlans_w_fat