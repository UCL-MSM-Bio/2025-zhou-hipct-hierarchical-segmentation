import os 
import glob
import json
from natsort import natsorted
import shutil
import tqdm
import shutil
from typing import Tuple

SAMPLE_LIST = ['5um S-20-28',
               '5.2um LADAF-2021-17 Left Kidney',
               '5.2um LADAF-2021-17 Right Kidney',
               '2.58um LADAF-2020-27 Left Kidney'
               ]

def gen_JSON(dataTr, dataTs, lblTr, save_path):
    data = {}
    data['name'] = 'Glomerulus'
    data['description'] = 'Glomerulus segmentation'
    data['Reference'] = 'UCL MSMaH Bio'
    data['licence'] = 'CC-BY-SA 4.0'
    data['release'] = '0.0.1'
    data['tensorImageSize'] = '3D'
    data['channel_names'] = {
        '0': 'HiPCT'
    }
    data['labels'] = {
        'background': 0,
        'glomerulus': 1
    }
    data['numTraining'] = len(dataTr)
    data['numTest'] = len(dataTs)
    data['file_ending'] = '.tif'
    #data['training'] = []
    #data['test'] = []

    # for image, label in zip(dataTr, lblTr):
    #     data['training'].append({'image': image, 'label': label})
    
    # for image in dataTs:
    #     data['test'].append(image)
    
    with open(os.path.join(save_path, 'dataset_test.json'), 'w') as f:
        json.dump(data, f, indent=4)

    return data

def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def processing_dataset(raw_data_path, raw_data_seperate_json, train_des_folder, train_label_des_folder, test_des_folder):
    with open(raw_data_seperate_json, 'r') as f:
        raw_data_seperate = json.load(f)
    train_data = []
    train_labels = []
    test_data = []
    for sample in SAMPLE_LIST:
        train_files = raw_data_seperate[sample]['train']['files']
        test_files = raw_data_seperate[sample]['test']['files']
        for item in train_files:
            train_labels.append(item)
            train_data.append(item.replace('label', 'cube'))
        for item in test_files:
            test_data.append(item.replace('label', 'cube'))
    natsorted(train_data)
    natsorted(train_labels)
    natsorted(test_data)

    spacing = (1, 1, 1)
    for i, (cube_name, lbl_name) in enumerate(zip(train_data, train_labels)):
        target_name = cube_name.split('.')[0] + f'_{i:03d}' # f'cube_image_{i:03d}'
        target_lbl_name = lbl_name.split('.')[0] + f'_{i:03d}' # f'cube_label_{i:03d}'
        # we still need the '_0000' suffix for images! Otherwise we would not be able to support multiple input
        # channels distributed over separate files
        original_cube_path = os.path.join(raw_data_path, 'cubes', cube_name)
        shutil.copy(original_cube_path, os.path.join(train_des_folder, target_name + '_0000.tif'))
        save_json({'spacing': spacing}, os.path.join(train_des_folder, target_name + '_0000.json'))

        original_label_path = os.path.join(raw_data_path, 'labels', lbl_name)
        
        if train_label_des_folder is not None:
            shutil.copy(original_label_path, os.path.join(train_label_des_folder, target_lbl_name.replace('label', 'cube') + '.tif'))
            # spacing file!
            save_json({'spacing': spacing}, os.path.join(train_label_des_folder, target_lbl_name.replace('label', 'cube') + '.json'))

    # test set, same a strain just without the segmentations
    for i, cube_name in enumerate(test_data):
        target_name = cube_name.split('.')[0] + f'_{i:03d}' 
        original_cube_path = os.path.join(raw_data_path, 'cubes', cube_name)
        shutil.copy(original_cube_path, os.path.join(test_des_folder, target_name + '_0000.tif'))
        # spacing file!
        save_json({'spacing': spacing}, os.path.join(test_des_folder, target_name + '_0000.json'))

    # now we generate the dataset json
    gen_JSON(
        dataTr=train_data,
        dataTs=test_data,
        lblTr=train_labels,
        save_path=os.path.join(raw_data_path, 'nnUNet_raw', 'Dataset001_Glomeruli')
    )
    # generate_dataset_json(
    #     os.path.join('D:\Yang\kidney_seg\\nnUNet\Dataset\\nnUnet_raw', 'Task01_Glomeruli'),
    #     {0: 'Glomerulus'},
    #     {'background': 0, 'glomerulus': 1},
    #     60,
    #     '.tif'
    # )


if __name__ == '__main__':

    '''Moving Images'''
    raw_data_path = '/hdd/yang/data/kidney_seg'
    raw_data_seperate_json = '/hdd/yang/data/kidney_seg/90_10/whole_data.json'
    train_des_folder = '/hdd/yang/data/kidney_seg/nnUNet_raw/Dataset001_Glomeruli/imagesTr_whole'
    train_label_des_folder = None #'/hdd/yang/data/kidney_seg/nnUNet_raw/Dataset001_Glomeruli/labelsTr'
    test_des_folder = '/hdd/yang/data/kidney_seg/nnUNet_raw/Dataset001_Glomeruli/imagesTs_whole'
    processing_dataset(raw_data_path, raw_data_seperate_json, train_des_folder, train_label_des_folder, test_des_folder)
    
    
    '''Prepare 5fold split'''
    # imageTr = glob.glob('/hdd/yang/data/kidney_seg/nnUNet_raw/Dataset001_Glomeruli/imagesTr/*.tif')
    # original_fold_path = '/hdd/yang/data/kidney_seg/90_10/valid_seperate_5fold.json'
    # nnUNet_fold_file_path = '/hdd/yang/data/kidney_seg/nnUNet_preprocessed/Dataset001_Glomeruli/splits_final.json'
    
    # all_training_files = {}
    # for file in imageTr:
    #     name1 = file.split('/')[-1]
    #     first_ind = name1.find('_0000')
    #     nnUNet_name = name1[:first_ind]
    #     second_ind = nnUNet_name.rfind('_')
    #     file_name = nnUNet_name[:second_ind]
    #     all_training_files[file_name] = nnUNet_name
    # #print(all_training_files)
    

    # nnUMet_fold = []
    # with open(original_fold_path, 'r') as f:
    #     original_fold = json.load(f)
    #     for fold in ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']:
    #         data = {'train': [], 'val': []}
    #         for item in original_fold[fold]['train']:
    #             item = item.split('.')[0].replace('label', 'cube')
    #             item = all_training_files[item]               
    #             data['train'].append(item)
    #         for item in original_fold[fold]['val']:
    #             item = item.split('.')[0].replace('label', 'cube')
    #             item = all_training_files[item]
    #             data['val'].append(item)
    #         nnUMet_fold.append(data)
    # save_json(nnUMet_fold, nnUNet_fold_file_path)
    # print(nnUMet_fold)


