import dask_image.imread
import numpy as np
from skimage import draw
import os

def loading_by_dask(image_path, crop_coords=None):
    '''
    Load images by dask_image.imread.imread
    Args:
        image_folder: path to image folder
        crop_z_range: crop z range (only for fixed image)
        moving_crop_coords: crop coordinates for moving image ROI (only for moving image) [top_left, bottom_right]
    '''
    if os.path.isdir(image_path):
        image_stack = dask_image.imread.imread(f'{image_path}/*.tif') # [slice, h, w]
    else:
        image_stack = dask_image.imread.imread(image_path)

    #if crop_z_range is not None:
    #    image_stack = image_stack[crop_z_range[0]:crop_z_range[1],:,:]
    if crop_coords is not None:
        image_stack = image_stack[crop_coords[0][2]:crop_coords[1][2], 
                                  crop_coords[0][1]:crop_coords[1][1], 
                                  crop_coords[0][0]:crop_coords[1][0]]
    image_array = image_stack.compute()
    return image_array