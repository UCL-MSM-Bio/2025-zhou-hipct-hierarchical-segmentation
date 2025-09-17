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