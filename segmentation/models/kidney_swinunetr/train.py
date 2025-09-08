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
from dataset import transforms
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

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.losses.tversky import TverskyLoss
from monai.metrics import DiceMetric
from monai.utils.misc import set_determinism
import wandb
from lr_scheduler import PolyLRScheduler

# def ddp_setup(rank: int, world_size: int):
#    """
#    Args:
#       rank: Unique identifier of each process
#       world_size: Total number of processes
#    """
#    os.environ["MASTER_ADDR"] = "localhost"
#    os.environ["MASTER_PORT"] = "12355"
#    init_process_group(backend="nccl", rank=rank, world_size=world_size)
#    torch.cuda.set_device(rank)

def setseed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def train_for_epoch(epoch, net, metricer, train_loader, loss_func, opt):
    # iteration_choice = tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9)
    iteration_choice = enumerate(train_loader)
    mean_loss = []
    dices = []
    for step, (images, labels) in iteration_choice:
        opt.zero_grad()
        images = images.cuda() #[b,1,32,256,256]
        labels = labels.long().cuda() #[b,32,256,256]

        predictions = net(images)  # [b,2,32,256,256]
        #print(predictions[0,:,0,0,0])
        #labels = labels.unsqueeze_(1) #[b,1,32,256,256]
        loss = loss_func(predictions, labels)
        mean_loss.append(loss.item())
        loss.backward()
        opt.step()
        
        
        # select the class with highest probability as prediction
        predictions_arg = predictions.argmax(dim=1).float()  # [b,32,256,256]
        #print('predictions_arg: ', predictions_arg.shape)
        labels = labels.squeeze(1) #[b,32,256,256]

        #metricer.set_input(labels, predictions_arg)
        #dices.append(metricer.dice_for_batch())
        
        metricer(predictions_arg, labels)

        if step % 100 == 0:
           print('epoch:{}, step:{}, loss:{:.3f}'.format(epoch, step, loss.item()))

    train_epoch_loss = sum(mean_loss) / len(mean_loss)
    train_epoch_dice = metricer.aggregate().item()
    metricer.reset()
    #train_epoch_dice = np.concatenate(dices,axis=0).mean(axis=0)[0] 
    #train_epoch_dice = np.mean(dices)
    print('train_loss:', train_epoch_loss, 'train_dice: ', train_epoch_dice)
    return train_epoch_loss, train_epoch_dice

def validatation_for_epoch(epoch, net, metricer, val_loader, loss_func):
    val_dices = []
    mean_loss = []
    # validation
    with torch.no_grad(): #
        for step, (image, label) in enumerate(val_loader):
            image = image.cuda() #[b,1,32,256,256]
            label = label.long().cuda() #[b,32,256,256]

            prediction = net(image)  # [b,2,32,256,256]
            #label = label.unsqueeze_(1) #[b,1,32,256,256]
            loss = loss_func(prediction, label)#, weight=weight2)

            mean_loss.append(loss.item())
            #if step %4 == 0:#for batch_size = 1 if not, remove it, probably no help
            # predictions_arg = predictions_stage2.argmax(dim=1).float()  # [b,32,256,256]
            prediction_arg = prediction.argmax(dim=1).float()  # [b,32,256,256]
            label = label.squeeze(1)
            #metricer.set_input(label, prediction_arg)
            #val_dices.append(metricer.dice_for_batch())
            metricer(prediction_arg, label)

    #val_epoch_dice = np.concatenate(val_dices,axis=0).mean(axis=0)[0]
    #val_epoch_dice = np.mean(val_dices)
    val_epoch_loss = sum(mean_loss) / len(mean_loss)
    val_epoch_dice = metricer.aggregate().item()
    metricer.reset()
    print('val_loss:', val_epoch_loss, 'val_dice: ', val_epoch_dice)
    return val_epoch_loss, val_epoch_dice

if __name__ == '__main__':

    setseed(0)

    # Logging
    wandb.init(
        project="kidney_seg",
        config={
            "model": config.MODEL_NAME,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.NUM_EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "decay_rate": config.DECAY_RATE,
            "lr_decay": config.LR_DECAY,
            "optimizer": config.OPTIMIZER,
            "fold": config.FOLD,
            "pre_trained": config.PRE_TRAINED,
        })
    print(wandb.config)

    # from monai
    net = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=1,
        out_channels=2,
        depths=(2,2,2,2),
        num_heads=(3,6,12,24),
        feature_size=24,
        norm_name="instance"
    )

    net = torch.nn.DataParallel(net).cuda()
    if config.PRE_TRAINED:
        print('Use pretrain model %s...'%config.PRE_TRAINED_MODEL)
        #logger.info('Use pretrain model')
        checkpoint = torch.load(os.path.join(config.SAVE_PATH, config.PRE_TRAINED_MODEL))
        start_epoch = checkpoint['epoch'] + 1
        glomeruli_dice = checkpoint['glomeruli_dice']
        net.load_state_dict(checkpoint['model_state_dict'])
        # net.load_state_dict(checkpoint)
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0
        glomeruli_dice = 0.0
        net = net.apply(weights_init)

    '''OPTIMIZER'''
    if config.OPTIMIZER == 'SGD':
        opt = torch.optim.SGD(net.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    elif config.OPTIMIZER == 'Adam':
        opt = torch.optim.Adam(
            net.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=config.DECAY_RATE
        )
    
    if config.PRE_TRAINED:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print('optimizer loaded')

    '''TRANSFORMS'''
    train_transforms = transforms.get_train_transform()
    val_transforms = transforms.get_val_transform()
    
    '''DATA LOADER'''
    data_path = config.DATA_PATH
    data_train = KidneyData(root=data_path, mode='train', seperation=config.SEPERATION, fold=config.FOLD, all_zero_involved=False, transforms=train_transforms)
    train_loader = DataLoader(data_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
    data_val = KidneyData(root=data_path, mode='val', seperation=config.SEPERATION, fold=config.FOLD, all_zero_involved=False, transforms=val_transforms)
    val_loader = DataLoader(data_val, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, drop_last=True)
    print('Loading data from {}, using fold {}, length of train set is {}, length of val set is {}'.format(data_path, config.FOLD, len(data_train), len(data_val)))



    '''LOSS FUNCTION'''
    # loss_func = DiceLoss(apply_softmax=True)
    loss_func = DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True)
    #loss_func = TverskyLoss(include_background=False, to_onehot_y=True, sigmoid=True, beta=0.7, alpha=0.3)
    #loss_func = torch.nn.NLLLoss()
    print('Loss function: ', loss_func)

    '''EVALUATION DEFINITION'''
    #metricer = Metric() #Metric()
    metricer = DiceMetric(include_background=False, reduction="mean")
    print('Metricer: ', metricer)

    '''LR SCHEDULER'''
    lr_scheduler = PolyLRScheduler(optimizer=opt, initial_lr=config.LEARNING_RATE, max_steps=config.NUM_EPOCHS, exponent=0.9, clip_lr=config.LEARNING_RATE_CLIP)
    print('LR scheduler: ', lr_scheduler)
    
    current_dice = glomeruli_dice
    print('Start_epoch: ', start_epoch)
    print('Current_dice: ', current_dice)
    # training process
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print('-----------------------Epoch %d-----------------------' % epoch)
        print('learning rate: ', opt.param_groups[0]['lr'])

        net.train()
        train_epoch_loss, train_epoch_dice = train_for_epoch(epoch, net, metricer, train_loader, loss_func, opt)

        net.eval()
        val_loss, val_dice = validatation_for_epoch(epoch, net, metricer, val_loader, loss_func)

        if val_dice > current_dice:
            current_dice = val_dice
            glomeruli_dice = current_dice
            state = {
                'epoch': epoch,
                'glomeruli_dice': glomeruli_dice,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }
            if not os.path.exists(config.SAVE_PATH):
                os.makedirs(config.SAVE_PATH)
            torch.save(state, os.path.join(config.SAVE_PATH, config.SAVE_MODEL_NAME+'.tar'))
            print('save model at epoch %d' % epoch)
        lr_scheduler.step(epoch)
        
        wandb.log({"train_loss": train_epoch_loss, 
                   "train_dice": train_epoch_dice, 
                   "val_loss": val_loss,
                   "val_dice": val_dice},
                   step=epoch)
        
    state_final = {'epoch': epoch,
                   'glomeruli_dice': val_dice,
                   'model_state_dict': net.state_dict(),
                   'optimizer_state_dict': opt.state_dict(),
                   }
    torch.save(state_final, os.path.join(config.SAVE_PATH, config.SAVE_MODEL_NAME+'_final.tar'))
        
