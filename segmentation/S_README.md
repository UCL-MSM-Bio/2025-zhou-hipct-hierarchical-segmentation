# Introduction
The segmentation folder involves the models used in this work. The models are listed:
```
├── models
│   ├── kidney_swinunetr
│   ├── kidney_unetr
│   ├── kidney_vnet
└── └── nnUNet
```
1. The VNet model is a fork from the [open-source repository](https://github.com/Flaick/VNet).
2. UNETR and SWINUNETR were implemented based on the VNet repository and [MONAI](https://monai.io/).
3. nnUNet V2 is a fork from the official [repository](https://github.com/MIC-DKFZ/nnUNet), version in late 2023. 

# Run the training on high-resolution data
The high-resolution data can be downloaded from [link](https://doi.org/10.5281/zenodo.15397768), where the data are the original 512x512x512 16-bit data.

## Data pre-processing
To prepare the manually annotated data for model training, the process includes (in order):

1. Convert the 512x512x512 16-bit data to 8-bit data using 3D CLAHE (Contrast Limited Adaptive Histogram Equalisation).
2. Divide the CLAHE cube into dimension of 128x128x128.
3. Select the training and testing cubes (original 512x512x512 cubes) across each sample (please see the paper).
4. Based on the selected training cubes, generate a .json file showing 5-fold cross-validation.

**VNet, UNETR and SwinUNETR can train 5-fold cross validation based on the .json file, while nnUNet requires further operations**

5. Prepare the data for nnUNet training and use the generated 5-fold cross-validation .json file as nnUNet default split file, so that all the models follow the same split.   

*nnUNetdata preprocess can be found in the original [repository](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).*

Please see the notebook at: *notebooks/preprocessing_manual_annot.ipynb* 

## Model training
For models of VNet, UNETR and SwinUNETR, there is a config.py for setting up the training hyperparameters. 

### VNet
```
uv run segmentation/models/kidney_vnet/train.py
```
### UNETR
```
uv run segmentation/models/kidney_unetr/train.py
```
### SwinUNETR
```
uv run segmentation/models/kidney_swinunetr/train.py 
```
### nnUNet
After processing the nnUNet_raw data completed, we can generate nnUNet_preprocessed data for the manual annotated data:
```
source .venv/bin/activate

# suppose the manual annotated dataset is 001
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# copy the split.json file to nnUNet
cp path/to/separation_5fold.json path/to/nnUNet_preprocessed_Dataset001_Glomeruli/split_final.json

# training dataset001 using fold 0
nnUNetv2_train 1 3d_fullres 0
```

# nnUNet fine-tuning on lower-resolution data
Since the nnUNet achieved the best performance on high-resolution annotated HiPCT data, we only applied nnUNet to fine-tune on lower-resolution data.

## Data pre-processing
Similar to pre-process high-resolution data, the process includes:

1. Apply CLAHE on 16-bit HiP-CT .jp2 original data.
2. Crop the training data according to the registered region. (The registration between higher-resolution to lower-resolution data can be found in the registration folder.)
3. Generate training cubes and prepare for nnUNet training (keep a percentage of empty cubes if needed).

See *notebooks/preprocessing_pseudo_lbls.ipynb* for an example.

## Fine-tuning
```
source .venv/bin/activate

# suppose the correlative pseudo-labeled dataset is 002
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity

# moving the training plan from last higher resolution. 
# suppose the higher-resolution dataset is 001 and current dataset is 002
nnUNetv2_move_plans_between_datasets -s 1 -t 2 -sp nnUNetPlans -tp nnUNetPlans_correlative

# after that, it is worthy checking the original generated nnUNetPlans (generated in step 1) and copy the intensity statistics to the nnUNetPLans_correlative.json


# prepare the dataset according to new Plans
nnUNetv2_preprocess -d 2 -plans_name nnUNetPlans_correlative

# Fine-tune using the model trained on higher-resolution data
nnUNetv2_train 2 3d_fullres 1 -p nnUNetPlans_correlative -pretrained_weights /nnUNet_results/Dataset001_Glomeruli/nnUNetTrainer__nnUNetPlans_w_fat__3d_fullres/fold_0/checkpoint_best.pth
```

To change the training epochs and learning rate, go to *2025-zhou-hipct-hierarchical-segmentation/segmentation/models/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py*, line 142 - 149
```
### Some hyperparameters for you to fiddle with

self.initial_lr = 2e-3 # 1e-2 default 2e-3
self.weight_decay = 3e-5
self.oversample_foreground_percent = 0.33
self.num_iterations_per_epoch = 250
self.num_val_iterations_per_epoch = 50
self.num_epochs = 1000 # 1000 default
self.current_epoch = 0
```
# Inference (Testing)
Test of VNet, UNETR and swinUNETR:

1. In the *test.py*, change the *test_data_path* in the main function. The *test_label_path* is set as **None** if test labels are not available.
2. Run the code

```
# test VNet
uv run segmentation/kidney_vnet/test.py

# test UNETR
uv run segmentation/kidney_unetr/test.py

# test swinUNETR
uv run segmentation/kidney_swinunetr/test.py
```

Test of nnUNet:
1. Prepare the training data as we did for training. Detailed see the original nnUNet [documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).
2. Run the inference code
```
source .venv/bin/activate

# high-resolution data with manual annotations
nnUNetv2_predict -i /input/folder/path -o /output/folder/path -d 1 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans

# lower-resolution data with pseudo-labels
nnUNetv2_predict -i /input/folder/path -o /output/folder/path -d 2 -c 3d_fullres -f 0 -step_size 0.5 -p nnUNetPlans_correlative

```