
import os
from time import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
#from torch.utils.data import DataLoader
from monai.data import DataLoader
from dataset.kidney_dataset import KidneyData
#from tqdm import tqdm
#from loss.Dice import DiceLoss_Focal, DiceLoss
#from common.Metric import Metric
import skimage.io as skio
import config 
# from self_attention_cv import UNETR
# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

from monai.networks.nets import UNETR
from monai.losses import DiceLoss
from monai.losses.tversky import TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, RandFlipd, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, RandShiftIntensityd, NormalizeIntensityd
from monai.utils.misc import set_determinism
import wandb
import glob
from natsort import natsorted
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer


def load_cubes_list(test_data_path, test_label_path=None, prediction_save_path=None):
    test_list = glob.glob(os.path.join(test_data_path, '*.tif'))
    test_list = [file for file in test_list]
    if prediction_save_path is not None:
        if not os.path.exists(prediction_save_path):
            os.makedirs(prediction_save_path)
    with open(os.path.join(prediction_save_path, 'log.txt'), 'w') as f:
        f.write('Test data path: {}\n'.format(test_data_path))
        f.write('Test data: {}\n'.format(len(test_list)))
    if test_label_path is not None:
        label_list = glob.glob(os.path.join(test_label_path, '*.tif'))
        label_list = [file for file in label_list]
        return natsorted(test_list), natsorted(label_list)
    return natsorted(test_list)

def load_cubes(cube_path, label_path=None, transform=None):
    cube = skio.imread(cube_path, plugin='tifffile')
    #cube = torch.FloatTensor(cube).unsqueeze(0).unsqueeze(0) # [1, 1, 32, 256, 256]
    #cube = transform(cube)
    if label_path is not None:
        label = skio.imread(label_path, plugin='tifffile')
        #label = torch.FloatTensor(label).unsqueeze(0) # [1, 32, 256, 256]
        data = transform({'image': cube, 'label': label})
        return data['image'].unsqueeze(0), data['label'].unsqueeze(0)
    data = transform({'image': cube})
    return data['image'].unsqueeze(0)
   

def load_model(checkpoint_path, model_name, prediction_save_path):
    with open(os.path.join(prediction_save_path, 'log.txt'), 'a') as f:
        f.write('Model path: {}\n'.format(checkpoint_path))
        f.write('Model name: {}\n'.format(model_name))
    checkpoint = torch.load(os.path.join(checkpoint_path, model_name))
    epoch = checkpoint['epoch']
    glomeruli_dice = checkpoint['glomeruli_dice']
    net = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 128),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0
    )
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(checkpoint['model_state_dict'])
    with open(os.path.join(prediction_save_path, 'log.txt'), 'a') as f:
        f.write('The model has been trained for {} epochs \nThe glomeruli dice is {:.4f}\n'.format(epoch, glomeruli_dice))
    return net

def predict_with_label(net, test_list, label_list, metricer, transform, prediction_save_path):
    net.eval()
    #metricer.reset()
    with torch.no_grad():
        for step, (image_path, label_path) in tqdm(enumerate(zip(test_list, label_list)), total=len(test_list)):
            cube, label = load_cubes(image_path, label_path, transform)
            cube = cube.cuda()
            label = label.long().cuda()
            pred = net(cube) 
            pred_args = pred.argmax(dim=1).float()

            metricer(pred_args, label)
            dice = dice = metricer.aggregate().item()
            metricer.reset()
            #metricer.set_input(label, pred_args)
            #dice = metricer.dice_for_batch()[0][0]

            pred_seg = pred_args.squeeze().cpu().numpy()
            pred_seg = pred_seg.astype(np.int8)
            saved_file_name = os.path.basename(image_path).replace('cube', 'pred')
            with open(os.path.join(prediction_save_path, 'log.txt'), 'a') as f:
                f.write('Dice for {}: {}\n'.format(saved_file_name, dice))
            skio.imsave(os.path.join(prediction_save_path, saved_file_name), pred_seg, plugin='tifffile', check_contrast=False)
    print('Prediction saved in {}'.format(prediction_save_path))

def predict_without_label(net, test_list, transform, prediction_save_path):
    net
    with torch.no_grad():
        for step, image_path in tqdm(enumerate(test_list)):
            cube = load_cubes(image_path, transform=transform)
            cube = cube.cuda()
            pred = net(cube) 
            pred_args = pred.argmax(dim=1).float() 
            pred_seg = pred_args.squeeze().cpu().numpy()
            pred_seg = pred_seg.astype(np.int8)
            saved_file_name = os.path.basename(image_path).replace('cube', 'pred')
            skio.imsave(os.path.join(prediction_save_path, saved_file_name), pred_seg, plugin='tifffile', check_contrast=False)
    print('Prediction saved in {}'.format(prediction_save_path))

def predict_sliding_window(net, test_list, transform, prediction_save_path):
    print('Sliding window prediction')
    net.eval()
    inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.5, mode='gaussian', sw_device='cuda', device='cpu')
    with torch.no_grad():
        for step, image_path in enumerate(tqdm(test_list)):
            cube = load_cubes(image_path, transform=transform)
            cube.to('cpu')
            pred = inferer(inputs=cube, network=net) # sliding windows
            pred_args = pred.argmax(dim=1).float() 
            pred_seg = pred_args.squeeze().cpu().numpy()
            pred_seg = pred_seg.astype(np.int8)
            saved_file_name = os.path.basename(image_path)
            segments = saved_file_name.split('_')
            saved_file_name = segments[0] + '_00_' + 'T0' + '_000.tif'
            skio.imsave(os.path.join(prediction_save_path, saved_file_name), pred_seg, plugin='tifffile', check_contrast=False)
    print('Prediction saved in {}'.format(prediction_save_path))

if __name__ == '__main__':
    # Load data list
    test_data_path = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset001_Glomeruli/imagesTr'
    test_label_path = '/hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset001_Glomeruli/labelsTr'
    # Other paths
    prediction_save_path = '/hdd/yang/results/glomeruli_segmentation/UNETR_results/fold2/validation/'

    if test_label_path is not None:
        test_list, label_list = load_cubes_list(test_data_path, test_label_path, prediction_save_path)
        print('Test data: {}, Test label: {}'.format(len(test_list), len(label_list)))
    else:
        test_list = load_cubes_list(test_data_path, test_label_path, prediction_save_path)
        print('Test data: {}'.format(len(test_list)))

    # Load model
    checkpoint_path = '/hdd/yang/results/glomeruli_segmentation/UNETR_results/fold2/'
    model_name = 'unetr_fold2.tar'
    net = load_model(checkpoint_path, model_name, prediction_save_path)
    
    # Metric
    metricer = DiceMetric(include_background=False, reduction="mean")

    # Transform
    transform = Compose([
        EnsureChannelFirstd(keys=['image'] ,channel_dim='no_channel'),
        #ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=['image']),
        ToTensord(keys=['image', 'label']),
    ])

    # Test
    if test_label_path is not None:
        predict_with_label(net, test_list, label_list, metricer, transform, prediction_save_path)
    else:
        #predict_without_label(net, test_list, transform, prediction_save_path)
        predict_sliding_window(net, test_list, transform, prediction_save_path)