

# Prepare dataset
Before training, extract dataset fingerprint(im size, voxel spacings, intensity information etc.) to build network structure
```Console
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
*DATASET_ID: dataset id without '00'*

Example:
```Console
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
```

# Move the plan from pre-trained model

Example
```Console
nnUNetv2_move_plans_between_datasets -s 5 -t 9 -sp nnUNetPlans_w_fat -tp nnUNetPlans_w_fat
```
-s: origin, would be the high-res dataset
-t: destination, would be new dataset
-sp: plan name on last resolution dataset
-tp: plan name on new dataset

**After that, it is worthy checking the original fingerprint and moved plans and replace the intensity statistics**

# Prepare data from new plan
Example
```Console
nnUNetv2_preprocess -d 5 -plans_name nnUNetPlans_w_fat
```

# Training / Finetune

```Console
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```
*DATASET_NAME_OR_ID: dataset id without '00'* \
*UNET_CONFIGURATION: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres*\
*FOLD: 0, 1, 2, 3, 4 (default 5-fold)*

Example:
```Console
nnUNetv2_train 5 3d_fullres 2 -p nnUNetPlans_w_fat -pretrained_weights /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset003_12-1Glom_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/checkpoint_best.pth

nnUNetv2_train 5 3d_fullres 2 -p nnUNetPlans_w_fat -tr nnUNetTrainer_WeightedCE -pretrained_weights /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset003_12-1Glom_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/checkpoint_best.pth

nnUNetv2_train 3 3d_fullres 1 -p nnUNetPlans_w_fat -pretrained_weights /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/checkpoint_best.pth

nnUNetv2_train 5 3d_fullres 1 -p nnUNetPlans_w_fat -pretrained_weights D:\Yang\results\glomeruli_segmentation\nnUNet_results\Dataset001_Glomeruli\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_1\checkpoint_best.pth

nnUNetv2_train 7 3d_fullres 3 -p nnUNetPlans_w_fat -pretrained_weights /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/checkpoint_best.pth

nnUNetv2_train 9 3d_fullres 1 -p nnUNetPlans_w_fat -pretrained_weights /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/checkpoint_best.pth

nnUNetv2_train 10 3d_fullres 0 -p nnUNetPlans_w_fat -pretrained_weights /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/checkpoint_best.pth
```


# Inference
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f FOLD --save_probabilities
```

Example:
```Console
nnUNetv2_predict -i /hdd/yang/data/kidney_seg/12.1um/full_8bits_tif/whole_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol -d 5 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat 

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/25.08um/full_8bits_tif/whole_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset004_25-08Glom_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol -d 4 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat 

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/25.08um/full_8bits_tif/training_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset005_25-08Glom_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_2/inference_whole_vol -d 5 -c 3d_fullres -f 2 -step_size 0.5 -p nnUNetPlans_w_fat

nnUNetv2_predict -i D:\Yang\data\kidney_seg\12.1um_data\whole_vol_patch -o D:\Yang\results\glomeruli_segmentation\nnUNet_results\Dataset005_12-1Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres\fold_0\inference_whole_col -d 5 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat

nnUNetv2_predict -i D:\Yang\data\kidney_seg\12.1um_data\training -o D:\Yang\results\glomeruli_segmentation\nnUNet_results\Dataset003_12-1Glom_w_fat\nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres\fold_0\inference_train_col -d 3 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat 

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/clahe_inference_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset009_25-08Glom_search_w_fat_label_filtered/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol -d 9 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/LADAF-2021-17_right/5.2um_LADAF-2021-17_kidney-right_VOI-02.1/clahe_inference_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/Inference_ladaf_2021_17_right_voi2-1 -d 1 -c 3d_fullres -f 1 -step_size 0.5 -p nnUNetPlans

```

nnUNetv2_predict -i /hdd/yang/data/kidney_seg_nnunet/nnUNet_test/Dataset007_25-08Glom_search_w_fat/fold0_imagesVal -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset007_25-08Glom_search_w_fat/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_on_validation -d 7 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat

nnUNetv2_predict -i /hdd/yang/data/kidney_seg_nnunet/nnUNet_test/Dataset009_25-08Glom_search_w_fat_label_filtered/fold0_imagesTr -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset009_25-08Glom_search_w_fat_label_filtered/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_on_training -d 9 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat

nnUNetv2_predict -i /hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/clahe_inference_vol -o /hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset009_25-08Glom_search_w_fat_label_filtered/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/inference_whole_vol -d 9 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_w_fat