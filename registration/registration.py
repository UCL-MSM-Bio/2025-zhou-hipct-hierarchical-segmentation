"""
Rewitten registration pipline for HiP-CT from Joseph Brunet's codes
"""
import registration_config as config
import helper
import glob
import natsort
import os
import loading
import skimage.io as skio
import numpy as np
import SimpleITK as sitk
import skimage as ski
import math
import matplotlib.pyplot as plt
import json

def display_cmpt_on_image(fixed_slice, moving_slice, fixed_cmpt, moving_cmpt):
    #fixed_image = ski.img_as_ubyte(fixed_slice)
    #moving_image = ski.img_as_ubyte(moving_slice)
    fixed_slice = sitk.GetArrayFromImage(fixed_slice)
    moving_slice = sitk.GetArrayFromImage(moving_slice)
    print(fixed_slice.shape, moving_slice.shape)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5)) 
    ax[0].imshow(fixed_slice, cmap='gray')
    ax[0].scatter(fixed_cmpt[0], fixed_cmpt[1], c='r', s=5)
    ax[0].set_title('Fixed image', fontsize=20)
    ax[0].set_axis_off()
    ax[1].imshow(moving_slice, cmap='gray')
    ax[1].scatter(moving_cmpt[0], moving_cmpt[1], c='r', s=5)
    ax[1].set_title('Moving image', fontsize=20)
    ax[1].set_axis_off()
    print('Fixed cmpt: ', fixed_cmpt)
    print('Moving cmpt: ', moving_cmpt)
    plt.show()

def prepare_data_dict(registration_cmpts, fixed_image_folder, moving_image_folder):
    fixed_info = {}
    fixed_info['res'] = float(registration_cmpts['low_res'].replace('um', '')) / 1000.0  # in mm
    fixed_info['cmpt'] =[int(math.floor(co)) for co in registration_cmpts['low_res_cmpt']]
    fixed_info['dir'] = fixed_image_folder
    moving_info = {}
    moving_info['res'] = float(registration_cmpts['high_res'].replace('um', '')) / 1000.0  # in mm
    moving_info['cmpt'] = [int(math.floor(co)) for co in registration_cmpts['high_res_cmpt']]
    moving_info['dir'] = moving_image_folder
    return fixed_info, moving_info

def pre_process(fixed_info, moving_info):
    """
    Pre-process the images:
        crop the z dimention for fixed image
        crop the round-shape ROI for moving image
    """
    new_fixed_info = fixed_info
    new_moving_info = moving_info

    fixed_images = natsort.natsorted(glob.glob(fixed_info['dir'] + '/*.tif'))
    fixed_images_num = len(fixed_images)
    fixed_shape = skio.imread(fixed_images[0]).shape
    fixed_shape = (fixed_shape[0], fixed_shape[1], fixed_images_num)
    moving_images = natsort.natsorted(glob.glob(moving_info['dir'] + '/*.tif'))
    moving_images_num = len(moving_images)
    moving_shape = skio.imread(moving_images[0]).shape
    moving_shape = (moving_shape[0], moving_shape[1], moving_images_num)
    
    moving_crop_coords = None
    if config.CROP_MOVING_ROI:
        moving_crop_coords, (cmpt_x_new, cmpt_y_new, cmpt_z_new) = helper.crop_moving_roi(moving_shape, moving_info)
        new_moving_info['cmpt'] = [cmpt_x_new, cmpt_y_new, cmpt_z_new]
    new_moving_info['crop_coords'] = moving_crop_coords
    #fixed_z_range = [0, fixed_images_num]
    fix_crop_coords = None
    if config.CROP_FIXED_ROI:
        fix_crop_coords, (cmpt_x_new, cmpt_y_new, cmpt_z_new)= helper.crop_fixed_roi(fixed_info, new_moving_info, moving_shape, fixed_shape, config.R_FACTOR)
        new_fixed_info['cmpt'] = [cmpt_x_new, cmpt_y_new, cmpt_z_new]
    new_fixed_info['crop_coords'] = fix_crop_coords

    print('Number of fixed images: {}, Shapes: {}'.format(fixed_images_num, fixed_shape))
    print('Number of moving images: {}, Shapes: {}'.format(moving_images_num, moving_shape))
    #print('Crop fix z: {} \n Fixed z range: {}'.format(config.CROP_FIXED_Z, fixed_z_range))
    print('Crop moving ROI: {} \n Moving crop coords (xyz): {}'.format(config.CROP_MOVING_ROI, moving_crop_coords))
    print('Crop fixed ROI: {} \n Fixed crop coords (xyz): {}'.format(config.CROP_FIXED_ROI, fix_crop_coords))
    print('Fixed commom points: {}'.format(new_fixed_info['cmpt']))
    print('Moving commom points: {}'.format(new_moving_info['cmpt']))

    return new_fixed_info, new_moving_info

def registration_rotation(fixed_image, moving_image, offset, rotation_centre, angle_range, angle_step, zrot):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsExhaustive(
        numberOfSteps=[0, 0, int((angle_range / 2) / angle_step), 0, 0, 0],
        stepLength=np.deg2rad(angle_step),
    )
    #R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=100)
    R.SetOptimizerScalesFromPhysicalShift()

    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)
    print('Initial transform:', theta_x, theta_y, theta_z, -offset)
    initial_transform = sitk.Euler3DTransform(rotation_centre, theta_x, theta_y, theta_z, -offset)
    # inPlace should be True, otherwise not runining
    R.SetInitialTransform(initial_transform, inPlace=True)
    final_transform = R.Execute(fixed_image, moving_image)
    print('Final metric value: ', R.GetMetricValue())
    print('Optimizer\'s stopping condition, ', R.GetOptimizerStopConditionDescription())
    print('Final transform: ', final_transform)
    return final_transform

def registration_itk(fixed_image, moving_image, offset, rotation_centre, zrot):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)

    R.SetMetricSamplingPercentage(0.01)

    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-7,
        numberOfIterations=2000,
        maximumNumberOfCorrections=20,
        maximumNumberOfFunctionEvaluations=2000,
        costFunctionConvergenceFactor=1e9,
    )

    R.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1]) # [4, 2, 2, 1, 1]
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0]) # [2, 2, 1, 1, 0]
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    w = 10
    R.SetOptimizerWeights([w, w, w, w, w, w, w / 1000])

    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)
    print('Initial transform:', theta_x, theta_y, theta_z, -offset)
    rigid_euler = sitk.Euler3DTransform(
        rotation_centre, theta_x, theta_y, theta_z, -offset
    )
    
    initial_transform = sitk.Similarity3DTransform()
    initial_transform.SetMatrix(rigid_euler.GetMatrix())
    initial_transform.SetTranslation(rigid_euler.GetTranslation())
    initial_transform.SetCenter(rigid_euler.GetCenter())

    del rigid_euler

    R.SetInitialTransform(initial_transform, inPlace=True)
    final_transform = R.Execute(fixed_image, moving_image)
    print('Final metric value: ', R.GetMetricValue())
    print('Optimizer\'s stopping condition, ', R.GetOptimizerStopConditionDescription())
    print('Final transform: ', final_transform)
    return final_transform

def resample(fix_image, moving_image, transform):
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    resampled = sitk.Resample(moving_image, fix_image, transform, interpolator, default_value)
    return resampled

def registration_pipline(fixed_info, moving_info, registration_res):

    fixed_info, moving_info = pre_process(fixed_info, moving_info)
    fixed_image_folder = fixed_info['dir']
    moving_image_folder = moving_info['dir']

    print('\n')
    print('Loading fixed images from {} \n moving image from {}'.format(fixed_image_folder, moving_image_folder))
    fixed_array = loading.loading_by_dask(fixed_image_folder, crop_coords=fixed_info['crop_coords'])
    moving_array = loading.loading_by_dask(moving_image_folder, crop_coords=moving_info['crop_coords'])
    print('Fixed image shape: {} \n Moving image shape: {}'.format(fixed_array.shape, moving_array.shape))

    # Setting up the image for registration
    fixed_image = sitk.GetImageFromArray(fixed_array)
    fixed_image.SetOrigin([0, 0, 0])
    fixed_spacing = fixed_info['res']
    fixed_image.SetSpacing([fixed_spacing, fixed_spacing, fixed_spacing])
    moving_image = sitk.GetImageFromArray(moving_array)
    moving_image.SetOrigin([0, 0, 0])
    moving_spacing = moving_info['res']
    moving_image.SetSpacing([moving_spacing, moving_spacing, moving_spacing])
    del fixed_array, moving_array

    normaliser = sitk.NormalizeImageFilter() # normalise the image to N(0,1)
    fixed_image = normaliser.Execute(fixed_image)
    moving_image = normaliser.Execute(moving_image)
    # simpleITK registration only works with float32 and float64
    pixelType = sitk.sitkFloat32
    fixed_image = sitk.Cast(fixed_image, pixelType)
    moving_image = sitk.Cast(moving_image, pixelType)
    
    display_cmpt_on_image(fixed_image[:,:,fixed_info['cmpt'][2]], moving_image[:,:,moving_info['cmpt'][2]], fixed_info['cmpt'], moving_info['cmpt'])
    print('Fixed image spacing: {} mm \n Moving image spacing: {} mm'.format(fixed_image.GetSpacing(), moving_image.GetSpacing()))
    print('Fixed image size: {} \n Moving image size: {}'.format(fixed_image.GetSize(), moving_image.GetSize()))

    # Calculate the offset and rotation centre
    fixed_cmpt = np.asarray(fixed_info['cmpt'])
    moving_cmpt = np.asarray(moving_info['cmpt'])
    trans_length = fixed_cmpt - (moving_cmpt * moving_spacing / fixed_spacing)
    offset = fixed_spacing * trans_length
    rotation_centre = fixed_cmpt * fixed_spacing
    print('Offset: {} mm \n Rotation centre: {} mm'.format(offset, rotation_centre))

    
    # 1. Rotation registration    
    print('\n')
    angle_range = 360
    angle_step = 2
    z_rot = 0
    print('Start the first Rotation Registration with: \n angle range: {} degree \n angle step: {} degree \n z rotation: {}'.format(angle_range, angle_step, z_rot))
    first_transformation = registration_rotation(fixed_image, moving_image, offset, rotation_centre, angle_range, angle_step, z_rot)
    print('First transformation: ', first_transformation)

    # 2. Fine Rotation registration
    print('\n')
    z_rot = np.rad2deg(first_transformation.GetParameters()[0:3][2])
    angle_range = 5
    angle_step = 0.1
    print('Start the second Rotation Registration with: \n angle range: {} degree \n angle step: {} degree \n z rotation: {}'.format(angle_range, angle_step, z_rot))
    second_transformation = registration_rotation(fixed_image, moving_image, offset, rotation_centre, angle_range, angle_step, z_rot)
    print('Second transformation: ', second_transformation)

    # 3. Translation registration 
    print('\n')
    z_rot = np.rad2deg(second_transformation.GetParameters()[0:3])[2]
    print('second zrot:', z_rot)  
    if z_rot < 0:
        z_rot += 360
    print('Final zrot:', z_rot)
    print('Start the Similarity Translation Registration with z rotation: {}'.format(z_rot))
    final_transform = registration_itk(fixed_image, moving_image, offset, rotation_centre, z_rot)
    print('Final transformation: ', final_transform)

    if config.SAVE_REGISTRATION:
        print('\n')
        resampled_image = resample(fixed_image, moving_image, final_transform)
        print('Convert to 8-bits...')
        resampled_image = helper.convert_16_to_8(resampled_image)
        fixed_image = helper.convert_16_to_8(fixed_image)

        #prefix = registration_res + '_slice_' + str(fixed_info['z_range'][0]) + '_' + str(fixed_info['z_range'][1])
        prefix = registration_res + 'memory_efficient'
        if config.CROP_MOVING_ROI:
            tfm_saved = prefix + '_crop' + '.tfm'
            moving_saved = registration_res + '_crop' + '.tif'
        else:
            tfm_saved = prefix + '.tfm'
            moving_saved = registration_res + '.tif'

        if not os.path.exists(config.SAVE_T_PATH):
            os.makedirs(config.SAVE_T_PATH)
        print('Saving the registration results to {} and {}'.format(config.SAVE_T_PATH, config.SAVE_RESULTS_PATH))
        sitk.WriteTransform(final_transform, os.path.join(config.SAVE_T_PATH,  tfm_saved))

        if not os.path.exists(config.SAVE_RESULTS_PATH):
            os.makedirs(config.SAVE_RESULTS_PATH)   
        skio.imsave(os.path.join(config.SAVE_RESULTS_PATH, moving_saved), sitk.GetArrayFromImage(resampled_image), plugin='tifffile', check_contrast=False)
        skio.imsave(os.path.join(config.SAVE_RESULTS_PATH, prefix) + '_fixed.tif', sitk.GetArrayFromImage(fixed_image), plugin='tifffile', check_contrast=False)
        # save the fix and moving info as json
        with open(os.path.join(config.SAVE_T_PATH, prefix) + '_fixed_info.json', 'w') as f:
            json.dump(fixed_info, f, indent=4)
        with open(os.path.join(config.SAVE_T_PATH, prefix) + '_moving_info.json', 'w') as f:
            json.dump(moving_info, f, indent=4)


if __name__ == "__main__":

    if config.JP2_TO_TIF:
        jp2_file_path_fixed = config.JP2_FILE_PATH_FIXED
        tif_save_path_fixed = config.TIF_SAVE_PATH_FIXED
        jp2_file_path_moving = config.JP2_FILE_PATH_MOVING
        tif_save_path_moving = config.TIF_SAVE_PATH_MOVING
        print('Transforming fixed images from jp2 to tif:')
        helper.jp2_to_tif(jp2_file_path_fixed, tif_save_path_fixed)
        print('Transforming moving images from jp2 to tif:')
        helper.jp2_to_tif(jp2_file_path_moving, tif_save_path_moving)
    
    registration_list_json = config.REGISTRATION_LIST
    registration_res = config.REGISTRATION_RES #config.HIGH_RES_PATH.split('um')[0].split('/')[-1] + '_reg_to_' + config.LOW_RES_PATH.split('um')[0].split('/')[-1]
    registration_cmpts = helper.read_cmpt_file(registration_list_json)
    registration_cmpts = registration_cmpts[registration_res]
    fixed_image_folder = config.FIXED_IM_PATH
    moving_image_folder = config.MOVING_IM_PATH
    helper.check_integrity(registration_cmpts, fixed_image_folder, moving_image_folder)
    fixed_info, moving_info = prepare_data_dict(registration_cmpts, fixed_image_folder, moving_image_folder)
    
    registration_pipline(fixed_info, moving_info, registration_res)
    print('\n DONE! \n')

    


    