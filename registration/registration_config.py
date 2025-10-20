# If input is jp2 format, Convert it to tif
JP2_TO_TIF = False
JP2_FILE_PATH_FIXED = None
JP2_FILE_PATH_MOVING = None
TIF_SAVE_PATH_FIXED = None
TIF_SAVE_PATH_MOVING = None

# check the registration list and select the correct registration resolutions
REGISTRATION_LIST = 'registration/registration_lists/LADAF-2020-27_kidney-left.json'
REGISTRATION_RES = '2.58_reg_to_12.1'

# FIXED = Low-res
FIXED_IM_PATH = "/hdd/yang/projects/glomeruli_segmentation/data/LADAF-2020-27-left/12.1um/original/normalised_3769/"
# MOVING = High-res
MOVING_IM_PATH = "/hdd/yang/projects/glomeruli_segmentation/data/LADAF-2020-27-left/2.58um/original/tif_5504/"

# crop the fixed z range to save the memory use in registration process
CROP_FIXED_ROI = True
CROP_MOVING_ROI = True
R_FACTOR = 1.1

# save the registration results
SAVE_REGISTRATION = True
SAVE_T_PATH = "registration/tfms/LADAF-2020-27_kidney-left_2.58_to_12.1"
SAVE_RESULTS_PATH = "registration/results/LADAF-2020-27_kidney-left_2.58um"
