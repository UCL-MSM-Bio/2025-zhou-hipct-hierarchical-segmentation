import numpy as np
import random
import skimage.io as skio
import numpy as np
import glob
import os
from natsort import natsorted
import json
from sklearn.model_selection import KFold
import shutil
from tqdm import tqdm

# Do not change the order. This is the sample order of the 40 cubes.
DATASETS = {'5um S-20-28': 20, 
            '5.2um LADAF 2021-17 Left Kidney':7,
            '5.2um LADAF 2021-17 Right Kidney':9,
            '2.58um LADAF 2020-27 Left Kidney':4
            }

def train_test_split(save_path, train_ratio=0.9, ):
    '''
    Split the 40 512x512x512 cubes into training and testing data based on a ratio.
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start = 0
    with open (f'{save_path}/train_selected_cubes_tr{train_ratio}.txt', 'w') as f_train:     
        with open (f'{save_path}/test_selected_cubes_tr{train_ratio}.txt', 'w') as f_test:
            for sample in DATASETS:
                print(f'Sample: {sample}, Total Cubes: {DATASETS[sample]}')
                end = start + DATASETS[sample]
                train_data_length = int(DATASETS[sample] * train_ratio)
                train_selected = random.sample(range(start, end), train_data_length)
                train_selected.sort()
                test_selected = list(set(range(start, end)) - set(train_selected))
                test_selected.sort()
                
                print(f'Train Indices: {train_selected}, total: {len(train_selected)}')
                print(f'Test Indices: {test_selected}, total: {len(test_selected)}')
                
                for i in range(len(train_selected)):
                    f_train.write(f'{train_selected[i]}\n')
                for i in range(len(test_selected)):
                    f_test.write(f'{test_selected[i]}\n')
                f_train.flush()
                f_test.flush()
                start = end

########
#
# The functions below are to generate a json file which contains the name of all the non-zero label cubes.
#
########

def all_zero(lb_file):
    '''
    Check if the label is all zero
    Args:
        lb_file: label file, matrix
    '''
    if np.sum(lb_file) == 0:
        return True
    else:
        return False

def load_label(path):
    '''
    Load .tif file which is generated from Amira with skimage.io
    Args:
        path: path of .tif file
    '''
    tif_stack = skio.imread(path, plugin='tifffile')
    #tif_stack = tifffile.imread(path)
    return tif_stack

def generate_patch_list_per_sample(label_dir, train_cubes_txt, test_cube_txt, output_path, non_zero_only=True):
    '''
    Save the samples with valid label into json file
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    seperation = {}
    seperation['train'] = [int(i) for i in open(train_cubes_txt).read().split('\n') if i != ''] 
    seperation['test'] = [int(i) for i in open(test_cube_txt).read().split('\n') if i != '']

    output_dict = {}
    output_dict_seperate = {}
    start = 0
    for sample in DATASETS.keys():
        print('Working on sample:', sample)
        end = start + DATASETS[sample]
        total_patch = []
        train_patch = []
        test_patch = []
        for cube_idx in range(start, end):
            lb_patch_list = natsorted(glob.glob(os.path.join(label_dir, 'label{}_*.tif'.format(cube_idx))))
            for lb_patch in lb_patch_list:
                lb = load_label(lb_patch)
                if not non_zero_only or not all_zero(lb):
                    total_patch.append(lb_patch.split('/')[-1])
                    if cube_idx in seperation['train']:
                        train_patch.append(lb_patch.split('/')[-1])
                    elif cube_idx in seperation['test']:
                        test_patch.append(lb_patch.split('/')[-1])
                
                      

        print('Total number of valid cubes:', len(total_patch))
        
        output_dict[sample] = {'amount':len(total_patch), 
                               'files': total_patch
                               }
        output_dict_seperate[sample] = {'train': {'amount':len(train_patch), 
                                                  'files': train_patch
                                                  },
                                        'test': {'amount':len(test_patch), 
                                                 'files': test_patch
                                                 }
                                        }
        start = end
    if non_zero_only:
        with open(os.path.join(output_path, 'total_patch_non_zero.json'), 'w') as f:
            json.dump(output_dict, f, indent=4)
        with open(os.path.join(output_path, 'non_zero_patch_list_per_sample.json'), 'w') as f:
            json.dump(output_dict_seperate, f, indent=4)
        print('Done!')
    else:
        with open(os.path.join(output_path, 'total_patch.json'), 'w') as f:
            json.dump(output_dict, f, indent=4)
        with open(os.path.join(output_path, 'patch_list_per_sample.json'), 'w') as f:
            json.dump(output_dict_seperate, f, indent=4)
        print('Done!')


def generate_folds(n_folds, patch_list_per_sample, output_json_dir):
    output_dict = {}
    for i in range(n_folds):
        output_dict['fold'+str(i)] = {'train':[], 'val':[]}
    with open(patch_list_per_sample, 'r') as f:
        data = json.load(f)
    for sample in data.keys():
        print('Working on sample:', sample)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        kf.get_n_splits(data[sample]['train']['files'])
        for i, (train_index, valid_index) in enumerate(kf.split(data[sample]['train']['files'])):
                output_dict['fold'+str(i)]['train'] += np.asarray(data[sample]['train']['files'])[train_index].tolist()
                output_dict['fold'+str(i)]['val'] += np.asarray(data[sample]['train']['files'])[valid_index].tolist()
                print('Train:', len(np.asarray(data[sample]['train']['files'])[train_index].tolist()))
                print('Val:', len(np.asarray(data[sample]['train']['files'])[valid_index].tolist()))

    for fold in output_dict.keys():
        print('Fold:', fold)
        print('Train:', len(output_dict[fold]['train']))
        print('Val:', len(output_dict[fold]['val']))

    with open(os.path.join(output_json_dir, f'seperation_{n_folds}folds.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)


################
#
# Prepare the data for nnUNet
#
################
def generate_nnunet_JSON(dataTr, save_path):
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
    data['file_ending'] = '.tif'
    data["overwrite_image_reader_writer"] = "Tiff3DIO"
    
    with open(os.path.join(save_path, 'dataset.json'), 'w') as f:
        json.dump(data, f, indent=4)

    return data

def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def processing_nnunet_dataset(patch_dir, patch_list_per_sample, nnunet_raw_data_dir, dataset_name):

    if not os.path.exists(nnunet_raw_data_dir):
        os.makedirs(nnunet_raw_data_dir)
    train_des_folder = os.path.join(nnunet_raw_data_dir, dataset_name, 'imagesTr')
    train_label_des_folder = os.path.join(nnunet_raw_data_dir, dataset_name, 'labelsTr')
    if not os.path.exists(train_des_folder):
        os.makedirs(train_des_folder)
    if not os.path.exists(train_label_des_folder):
        os.makedirs(train_label_des_folder)
    
    with open(patch_list_per_sample, 'r') as f:
        raw_data_seperate = json.load(f)
    train_data = []
    train_labels = []
    for sample in DATASETS.keys():
        train_files = raw_data_seperate[sample]['train']['files']
        for item in train_files:
            train_labels.append(item)
            train_data.append(item.replace('label', 'cube'))
    natsorted(train_data)
    natsorted(train_labels)

    spacing = (1, 1, 1)
    # using tqdm for emumerate
    
    for i, (cube_name, lbl_name) in enumerate(tqdm(zip(train_data, train_labels), total=len(train_data), desc="Copying training data")):
        target_name = cube_name.split('.')[0] + f'_{i:03d}' # f'cube_image_{i:03d}'
        target_lbl_name = lbl_name.split('.')[0] + f'_{i:03d}' # f'cube_label_{i:03d}'
        # we still need the '_0000' suffix for images! Otherwise we would not be able to support multiple input
        # channels distributed over separate files
        original_cube_path = os.path.join(patch_dir, 'cubes', cube_name)
        shutil.copy(original_cube_path, os.path.join(train_des_folder, target_name + '_0000.tif'))
        save_json({'spacing': spacing}, os.path.join(train_des_folder, target_name + '.json'))

        original_label_path = os.path.join(patch_dir, 'labels', lbl_name)
        shutil.copy(original_label_path, os.path.join(train_label_des_folder, target_lbl_name.replace('label', 'cube') + '.tif'))
        # spacing file!
        save_json({'spacing': spacing}, os.path.join(train_label_des_folder, target_lbl_name.replace('label', 'cube') + '.json'))

    # now we generate the dataset json
    dataset_dir = os.path.join(nnunet_raw_data_dir, dataset_name)
    generate_nnunet_JSON(
        dataTr=train_data,
        save_path=dataset_dir
    )