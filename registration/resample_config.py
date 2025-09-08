# Image Paths - FIXED (low-res) and MOVING (high-res)
FIXED_IM_PATH = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/25.08um/normalised'
MOVING_IM_PATH = '/hdd/yang/data/kidney_seg/LADAF-2020-27_left/2.58um/normalised'
FIXED_RES = 25.08
MOVING_RES = 2.58

REGISTRATION_LIST = '/hdd/yang/code/hipct_registration/registration_lists/LADAF-2020-27_kidney-left.json'
REGISTRATION_RES = '2.58_reg_to_25.08'
MOVING_SQAURE_CROP = False

# type: image or label. Image is for CT scan, label is for masks
R_TYPE = 'image'
T_PATH = '/hdd/yang/code/hipct_registration/tfms/LADAF-2020-27_kidney-left_2.58_to_25.08'
SAVE_PATH = '/hdd/yang/code/hipct_registration/results/LADAF-2020-27_kidney-left_2.58um'

SAVE_IMAGE = False
