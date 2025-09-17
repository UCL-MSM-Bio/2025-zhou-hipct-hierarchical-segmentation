import os
# add parent path
import glob
import numpy as np
import skimage.io as skio
import torch
from torch.utils.data import Dataset
import random
from natsort import natsorted
import json

class KidneyData(Dataset):
    def __init__(self, root='data/', mode='train', separation='90_10', fold=0, all_zero_involved=False, transforms=None):
        super(KidneyData, self).__init__()
        self.root = root
        self.mode = mode
        self.all_zero_involved = all_zero_involved
        self.transforms = transforms
        self.fold = fold
        self.image_list=[]
        self.label_list=[]

        try:
            fold_file = os.path.join(self.root, separation, 'separation_5folds.json')
        except FileNotFoundError:
            raise FileNotFoundError('Please generate the k-fold json file first!')
        
        with open(fold_file, 'r') as f:
            fold_dict = json.load(f)
        
        self.label_list = fold_dict[self.fold][self.mode]
        for item in self.label_list:
            self.image_list.append(item.replace('label', 'cube'))
        
        self.image_list = natsorted(self.image_list)
        self.label_list = natsorted(self.label_list)

        # if self.mode == 'train':
        #     data_path = os.path.join(self.root, 'train')
        # elif self.mode == 'val':
        #     data_path = os.path.join(self.root, 'val')
        # else:
        #     raise ValueError('Wrong mode!')
        # print('Loading {} data from {}'.format(mode, data_path))
        # print('transforms applied: {}'.format(self.transforms))

        # if self.all_zero_involved:
        #     print('All zero cubes are involved!')
        #     #self.image_list = natsorted(glob.glob(os.path.join(data_path, 'images', '*.tif')))
        #     #self.label_list = natsorted(glob.glob(os.path.join(data_path, 'labels', '*.tif')))
        #     raise ValueError('Not implemented yet!')
        # else:
        #     print('Train on non-zero cubes only')
        #     with open(os.path.join(data_path, 'valid_label.txt'), 'r') as f:
        #         for line in f.readlines():
        #             file_name = line.split('\n')[0]
        #             self.image_list.append(os.path.join(data_path, 'images', file_name.replace('label', 'cube')))
        #             self.label_list.append(os.path.join(data_path, 'labels', file_name))
        #     self.image_list = natsorted(self.image_list)
        #     self.label_list = natsorted(self.label_list)
            
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root, 'cubes', self.image_list[index])
        label_path = os.path.join(self.root, 'labels', self.label_list[index])
        # print(image_path)
        # print(label_path)
        image = skio.imread(image_path, plugin='tifffile')
        label = skio.imread(label_path, plugin='tifffile')
        if self.transforms is not None:
            data = self.transforms({'image': image, 'label': label})
            image = data['image'].astype(torch.float) #.unsqueeze(0) # [1, 32, 256, 256]
            label = data['label'].astype(torch.float)  #.unsqueeze(0) # [32, 256, 256]
        
        #label = label/255.0
        return image, label
        
if __name__ == '__main__':
    np.random.seed(0)
    dataset = KidneyData(root='D:\Yang\data\kidney_seg', mode='val', separation='90_10', fold=0)
    print(len(dataset))
    #image, label = dataset[0]
    #print(image.shape)
    #print(label.shape)