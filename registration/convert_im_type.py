import SimpleITK as sitk
import numpy as np
import dask_image.imread
import skimage.io as skio

def convert_16_to_8(image_path, save_path):
    im_arr_16 = dask_image.imread.imread(image_path)
    im_sitk= sitk.GetImageFromArray(im_arr_16)
    im_sitk_8 = sitk.Cast(sitk.RescaleIntensity(im_sitk), sitk.sitkUInt8)
    im_arr_8 = sitk.GetArrayFromImage(im_sitk_8)
    if save_path != 'none':
        skio.imsave(save_path, im_arr_8, plugin='tifffile', check_contrast=False)

if __name__ == '__main__':
    image_path = 'results/2.58_reg_to_12.1.tif'
    save_path = 'results/2.58_reg_to_12.1_8.tif'
    convert_16_to_8(image_path, save_path)