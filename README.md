# Multiscale Segmentation using HiP-CT

This is the official repository for the publication entitled **Multiscale Segmentation using Hierarchical Phase-contrast Tomography and Deep Learning**

# Folder Structure

```
	├── LICENSE
	├── main.py
	├── notebooks # Example notebooks
	├── pyproject.toml # Python environment requirement
	├── README.md
	├── registration
	│   ├── LICENSE
	│   ├── README.md
	│   ├── registration_lists # Common points for the registration
	│   └── tfms # Registration transformations
	├── segmentation
	│   ├── models # Models used 
	│   ├── postprocessing
	│   └── preprocessing
	└── uv.lock
```

# Contents
1. [Segmentation](/segmentation/S_README.md)
2. [Registration](/registration/R_README.md)

# Python environment configuration
To set up the environment, [uv](https://docs.astral.sh/uv/) is used:
```
git clone https://github.com/UCL-MSM-Bio/2025-zhou-hipct-hierarchical-segmentation.git
cd 2025-zhou-hipct-hierarchical-segmentation
uv sync

# after that, configure the nnUNet environemnt (not integrated in uv)
source .venv/bin/activate
cd segmentation/models/nnUNet
pip install -e .
```
If you prefer using conda, please install the packages in the *pyproject.toml*, but you are still required to install nnUNet using pip (see the nnUNet official installation [document](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)).

# Data Availability
The high-resolution manually annotated data (in 2.58 um/voxel ~ 5.2 um/voxel) are available at https://doi.org/10.5281/zenodo.15397768.

The complete HiP-CT kidney volumes are available at [HOAHub Portal](https://human-organ-atlas.esrf.fr/). The DOIs are in the manuscript.

# Citation

```
 @article {Zhou2025.05.15.654263,
	author = {Zhou, Yang and Aslani, Shahab and Javanmardi, Yousef and Brunet, Joseph and Stansby, David and Carroll, Saskia and Bellier, Alexandre and Ackermann, Maximilian and Tafforeau, Paul and Lee, Peter D. and Walsh, Claire L.},
	title = {Multiscale Segmentation using Hierarchical Phase-contrast Tomography and Deep Learning},
	elocation-id = {2025.05.15.654263},
	year = {2025},
	doi = {10.1101/2025.05.15.654263},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/05/27/2025.05.15.654263},
	eprint = {https://www.biorxiv.org/content/early/2025/05/27/2025.05.15.654263.full.pdf},
	journal = {bioRxiv}
}
```