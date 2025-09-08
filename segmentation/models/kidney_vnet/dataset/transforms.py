import numpy as np
from monai import transforms

class ZScoreNormalisation(object):
    '''
    Calculate the z-score normalisation of the input image in a batch, i.e. (x-mean)/std.
    '''
    def __init__(self):
        self.eps = 1e-8
    def __call__(self, sample):
        mean = sample.mean()
        std = sample.std()
        sample = (sample - mean) / (max(std, 1e-8))
        return sample
    def __repr__(self):
        return self.__class__.__name__
    
class CTNormalisation(object):
    '''
    Similar to the nnUNet class
    '''
    def __init__(self, mean, std, percentile_00_5, percentile_99_5):
        self.mean = mean
        self.std = std
        self.lower_bound = percentile_00_5
        self.upper_bound = percentile_99_5
    
    def __call__(self, sample):
        sample = np.clip(sample, self.lower_bound, self.upper_bound)
        sample = (sample - self.mean) / (max(self.std, 1e-8))
        return sample

def get_train_transform():
    tfms = transforms.Compose([
        transforms.EnsureChannelFirstd(keys=['image', 'label'] ,channel_dim='no_channel'),
        transforms.RandRotated(keys=['image', 'label'], range_x=0.5235987755982988, range_y=0.5235987755982988, range_z=0.5235987755982988, prob=0.2),
        transforms.RandZoomd(keys=['image', 'label'], min_zoom=0.7, max_zoom=1.4, prob=0.2),
        transforms.RandGaussianNoised(keys=['image'], prob=0.1),
        transforms.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        transforms.RandShiftIntensityd(keys=['image'], prob=0.15, offsets=(0.75, 1.25)),
        transforms.RandAdjustContrastd(keys=['image'],prob=0.15, gamma=(0.75, 1.25)),
        transforms.RandFlipd(keys=['image', 'label'],prob=0.5, spatial_axis=(0,1,2)),
        transforms.NormalizeIntensityd(keys=['image']),
        transforms.ToTensord(keys=['image', 'label']),
    ])
    display_transform(tfms, 'Train_transforms')
    return tfms

def get_val_transform():
    tfms = transforms.Compose([
        transforms.EnsureChannelFirstd(keys=['image', 'label'] ,channel_dim='no_channel'),
        transforms.NormalizeIntensityd(keys=['image']),
        transforms.ToTensord(keys=['image', 'label']),
    ])
    display_transform(tfms, 'Val_transforms')
    return tfms

def display_transform(tfms, discription='Train_transforms'):
    print(discription + ': ', end='')
    tfms_list = []
    for transform in tfms.transforms:
        transform = str(transform)
        transform = transform.split(' ')[0].split('.')[-1]
        tfms_list.append(transform)
    print(tfms_list)


if __name__ == '__main__':
    from kidney_dataset import KidneyData
    from torch.utils.data import DataLoader
    data_path = 'D:\Yang\data\kidney_seg'
    # train test
    train_tfms = get_train_transform()
    data_train = KidneyData(root=data_path, mode='train', transforms=train_tfms)
    train_loader = DataLoader(data_train, batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    print(len(data_train))
    print(len(train_loader))
    # val test
    val_tfms = get_train_transform()
    data_val = KidneyData(root=data_path, mode='val', transforms=val_tfms)
    val_loader = DataLoader(data_train, batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    print(len(data_val))
    print(len(val_loader))