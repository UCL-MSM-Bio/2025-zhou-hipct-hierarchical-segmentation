# Introduction
The registration folder involves the multiscale HiP-CT registration for pseudo-labelling. The files in the folder are structured as follows:
```
├── registration_lists/ # the common points for the kidneys in the paper
├── tfms/ # the registration transformations saved folder
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

# Resample (pseudo-labelling)
1. Set up the resample configurations in resample_config.py
- MOVING_IM_PATH can be set to the post-processed binary prediction
- When the MOVING_IM_PATH is a single binary prediction .tif file, set the R_TYPE as 'label'
