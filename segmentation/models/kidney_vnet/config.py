import os 

# Model
MODEL_NAME = 'vnet'

# wandb logging
WANDB_LOG = False

# Classes
ORGANS = ['glomeruli']
NUM_CLASSES = len(ORGANS) + 1 # +1 for background

# Lab Machine
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Project path: ", PROJECT_PATH)
DATA_PATH = '/hdd/yang/projects/glomeruli_segmentation/2025-zhou-hipct-hierarchical-segmentation/data/high-res_training'

# CS Cluster
# PROJECT_PATH = '/home/zhouyang/kidney_vnet'
# DATA_PATH = '/home/zhouyang/data/kidney_seg'

# Dataloader path for n-fold cross validation
SEPARATION='90_10'
FOLD=3

SAVE_PATH = os.path.join(PROJECT_PATH, 'saves')
SAVE_MODEL_NAME = MODEL_NAME + '_fold' + str(FOLD)

# model-fold-res-hpc.tar

# Pretrained model
PRE_TRAINED = False
PRE_TRAINED_MODEL = MODEL_NAME + '_fold' + str(FOLD) + '_final.tar'

# Training parameters
BATCH_SIZE = 2
NUM_EPOCHS = 150
LEARNING_RATE = 0.0002
DECAY_RATE = 1e-4
AUGMENT = True
STEP_SIZE = 50
LR_DECAY = 0.7
NUM_WORKERS = 4
LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECAY = 0.5
MOMENTUM_DECAY_STEP = 50

USE_GPU = True
OPTIMIZER = 'Adam'
PRE_TRAINED_PATH = None