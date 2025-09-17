import os 

# Model
MODEL_NAME = 'swin-unetr'

# Classes
ORGANS = ['glomeruli']
NUM_CLASSES = len(ORGANS) + 1 # +1 for background

# CZI Path
PROJECT_PATH = 'kidney_swinunetr'
DATA_PATH = '/hdd/yang/projects/glomeruli_segmentation/data/high-res_training/'

# CS Cluster
# PROJECT_PATH = '/home/zhouyang/kidney_vnet'3
# DATA_PATH = '/home/zhouyang/data/kidney_seg'

# JADE2
# PROJECT_PATH = 'kidney_swinunetr'
# DATA_PATH = 'data/kidney_seg'

# Dataloader path for n-fold cross validation
SEPARATION='90_10'
FOLD=4

SAVE_PATH = os.path.join(PROJECT_PATH, 'saves')
SAVE_MODEL_NAME = MODEL_NAME + '_fold' + str(FOLD)

# Pretrained model
PRE_TRAINED = False
PRE_TRAINED_MODEL = MODEL_NAME + '_fold' + str(FOLD) + '_final.tar'

# Training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 150
LEARNING_RATE = 0.0005
DECAY_RATE = 1e-4
AUGMENT=True
STEP_SIZE = 50
LR_DECAY = 0.7
NUM_WORKERS = 4
LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = 50

USE_GPU = True
OPTIMIZER = 'Adam'
PRE_TRAINED_PATH = None