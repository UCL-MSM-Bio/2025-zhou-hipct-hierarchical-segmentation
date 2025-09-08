# If input is jp2 format, Convert it to tif
JP2_TO_TIF = False
JP2_FILE_PATH_FIXED = None
JP2_FILE_PATH_MOVING = None
TIF_SAVE_PATH_FIXED = None
TIF_SAVE_PATH_MOVING = None

# check the registration list and select the correct registration resolutions
REGISTRATION_LIST = '/hdd/yang/code/hipct_registration/registration_lists/LADAF-2020-27_kidney-left.json'
REGISTRATION_RES = '2.58_reg_to_25.08'

# FIXED = Low-res
FIXED_IM_PATH = "/hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/normalised"
# MOVING = High-res
MOVING_IM_PATH = "/hdd/yang/data/kidney_seg/LADAF-2020-27_left/2.58um/normalised"

# crop the fixed z range to save the memory use in registration process
CROP_FIXED_ROI = True
CROP_MOVING_ROI = True
R_FACTOR = 1.1

# save the registration results
SAVE_REGISTRATION = True
SAVE_T_PATH = "tfms/LADAF-2020-27_kidney-left_2.58_to_25.08"
SAVE_RESULTS_PATH = "results/LADAF-2020-27_kidney-left_2.58um"
