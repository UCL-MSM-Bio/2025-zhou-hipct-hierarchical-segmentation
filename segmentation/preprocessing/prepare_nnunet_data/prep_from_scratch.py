import json
import os
import numpy as np
import shutil
import skimage.io as skio

def generate_training_data(im_path, lbl_path, crop_save_path, patch_size, overlap=0.5,):
    im_save_path = os.path.join(crop_save_path, 'imagesTr')
    lbl_save_path = os.path.join(crop_save_path, 'labelsTr')
    im = skio.imread(im_path, plugin='tifffile')
    lbl = skio.imread(lbl_path, plugin='tifffile')
    assert im.shape == lbl.shape, 'Image and label shape mismatch'
    z, y, x = im.shape
    print('Image shape:', im.shape)
    z_stride = int(patch_size[0] * (1-overlap))
    y_stride = int(patch_size[1] * (1-overlap))
    x_stride = int(patch_size[2] * (1-overlap))
    json_dict = {
    "spacing": [
        1,
        1,
        1
        ]
        }
    print(f'Stride z: {z_stride}, y: {y_stride}, x: {x_stride}')
    print('Generating patches...')
    count = 0
    for i in range(0, z, z_stride):
        if i + patch_size[0] > z:
            break
        for j in range(0, y, y_stride):
            if j + patch_size[1] > y:
                break
            for k in range(0, x, x_stride):
                if k + patch_size[2] > x:
                    break
                patch_im = im[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
                patch_lbl = lbl[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
                patch_name = f'cube_z{i}_y{j}_x{k}_{count:03}_0000.tif'
                label_name = f'cube_z{i}_y{j}_x{k}_{count:03}.tif'
                json_name = f'cube_z{i}_y{j}_x{k}_{count:03}.json'
                skio.imsave(os.path.join(im_save_path, patch_name), patch_im, plugin='tifffile', check_contrast=False)
                skio.imsave(os.path.join(lbl_save_path, label_name), patch_lbl, plugin='tifffile', check_contrast=False)
                with open(os.path.join(im_save_path, json_name), 'w') as f:
                    json.dump(json_dict, f)
                with open(os.path.join(lbl_save_path, json_name), 'w') as f:
                    json.dump(json_dict, f)
                count += 1
    print('Total patches:', count)



if __name__ == '__main__':
    # load the plan
    nnUNet_plan_path = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_preprocessed/Dataset008_25-08Glom_search_whole/nnUNetPlans.json'
    nnUNet_plan = json.load(open(nnUNet_plan_path))
    patch_size = nnUNet_plan['configurations']['3d_lowres']['patch_size']
    print('Generated from patch size:', patch_size)

    # load the data
    raw_data_folder = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset008_25-08Glom_search_whole'
    preprocessed_data_folder = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_preprocessed/Dataset008_25-08Glom_search_whole'
    for folder in ['imagesTr', 'labelsTr']:
        shutil.rmtree(os.path.join(raw_data_folder, folder), ignore_errors=True)
        os.makedirs(os.path.join(raw_data_folder, folder))
    for folder in ['imagesTr', 'labelsTr', 'gt_segmentations', 'nnUNetPlans_2d', 'nnUNetPlans_3d_fullres', 'nnUNetPlans_3d_lowres']:
        shutil.rmtree(os.path.join(preprocessed_data_folder, folder), ignore_errors=True)
        os.makedirs(os.path.join(preprocessed_data_folder, folder))
    
    # copy the data
    training_data_path = '/hdd/yang/data/kidney_seg/25.08um/training/training_vol_roi/low_res_vol_roi.tif'
    training_label_path = '/hdd/yang/data/kidney_seg/25.08um/training/training_vol_roi/registered_lbl_roi.tif'
    crop_save_path = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset008_25-08Glom_search_whole'
    generate_training_data(training_data_path, training_label_path, crop_save_path, patch_size=patch_size, overlap=0)

