# Introduction
The registration folder involves the multiscale HiP-CT registration for pseudo-labelling. The files in the folder are structured as follows:
```
├── registration_lists/ # the common points for the kidneys in the paper
├── tfms/ # the registration transformations saved folder
├── correlation/ # the registration transformations saved folder
├── registration_config.py # registration inputs
├── registration.py # registration code
├── resample_config.py # resample inputs
└── resample.py # resample code   
```

# Registration process
1. Registration start from the manual picked common points as shown in the Figure below.
- The common point coordinates are in a format of (x, y, z), using [ImageJ](https://imagej.net/software/fiji/downloads)
- Fill the common points to the folder *registration_lists/*
  
  ![common points](figure_readme/github_reg_repo.png)

2. Edit the registration_config.py with the image paths, resolutions etc.
3. Before registration, run the normalisation to convert the 16-bit .jp2 image into 8-bit tif image. See the */notebooks/preprocessing_pseudo_lbls.ipynb* Normalise for Registration section. 
4. Run the registration!
```
uv run registration/registration.py
```

# Resample
1. Set up the resample configurations in resample_config.py
- MOVING_IM_PATH can be set to the post-processed binary prediction
- When the MOVING_IM_PATH is a single binary prediction .tif file, set the R_TYPE as 'label'

# Pseudo-labelling
After resampling the predictions from higher-resolution data to lower-resolution data, as shown in the figure below, the cropping coordicates of each axies can be decided by using [napari](https://napari.org/stable/). The coordicates used in this work are recorded in the *generate_training_column.py*. To generate the training data with pseudo-labels:
1. Apply the CLAHE on the lower-resolution data, example can be found in *notebook/preprocesssing_pseudo_lbls.ipynb*.
2. Crop the registered column based on the coordinates found using napari (or other tools).
3. Run the script to crop the column
```
# Before running, set up the paths in the __main__ 

uv run registration/correlation/generate_training_column.py
```
4. Generate nnUNet training data
```
uv run registration/correlation/correlative_nnunet_dataset.py
```

![pseudo-labelling](figure_readme/pseudo-labelling.png)

