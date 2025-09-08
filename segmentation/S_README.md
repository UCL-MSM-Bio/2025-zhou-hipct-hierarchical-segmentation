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
The high-resolution data can be downloaded from [link](https://doi.org/10.5281/zenodo.15397768), where the data are the original 512x512x512 16-bit data. To prepare the data in 128x128x128 and apply CLAHE (Contrast Limited Adaptive histogram Equalization), please see the notebook at: *notebooks/preprocessing_manual_annot.ipynb* 

## VNet
```
uv run segmentation/models/kidney_vnet/train.py
```
## UNETR
```
uv run segmentation/models/kidney_unetr/train.py
```
## SwinUNETR
