import skimage.io as skio
import numpy as np
import napari
import tifffile
import argparse

def load_image(path):
    '''
    Load .tif file which is generated from Amira with skimage.io
    Args:
        path: path of .tif file
    '''
    tif_stack = skio.imread(path, plugin='tifffile')
    #tif_stack = tifffile.imread(path)
    return tif_stack

def load_label(path, type=0):
    '''
    Load .label file which is generated from Amira with skimage.io
    Args:
        path: path of .label file
        type: 0 gt, 1 pred
    '''
    tif_stack = skio.imread(path, plugin='tifffile')
    if type == 1:
        tif_stack = tif_stack*2
    return tif_stack

def display(img_stack=None, lbl_stack=None, lbl_pred_stack=None):
    '''
    Display the label with napari
    Args:
        tif_stack: label to be displayed, [512, 512, 512]
    '''
    viewer = napari.Viewer()
    if img_stack is not None:
        viewer.add_image(img_stack)
    if lbl_stack is not None:
        viewer.add_labels(lbl_stack, name='label')
    if lbl_pred_stack is not None:
        viewer.add_labels(lbl_pred_stack, name='pred')
    napari.run()
    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('num', type=str, default=1, help='image to show')
    #args = parser.parse_args()

    #img_path = 'data/tif_cubes/cube'+args.num+'_T0.tif'
    #lbl_path = 'data/tif_labels_no_death/label'+args.num+'_T0.tif'
    img_path = 'D:\Yang\data\kidney_seg\\val\images\\cube1_25_T0.tif'
    lbl_path = 'D:\Yang\data\kidney_seg\\val\labels\\label1_25_T0.tif'
    lbl_pred_path = 'E:\Kidnet_seg\prediction\\unetr\\test_on_val-b8_jade2-280923\\pred1_25_T0.tif'
    # lbl_pred_path = 'D:\Yang\kidney_seg\\nnUNet\Dataset\\nnUnet_trained_models\Dataset002_Glomeruli\\nnUNetTrainer__nnUNetPlans__3d_fullres\\fold_5\\validation \
    #               \\cube15_45_T0_T0.tif'
    #lbl_pred_path = 'E:\Kidnet_seg\prediction\\nnUNet\cube41_53_T0_T0.tif'
    print(img_path)
    print(lbl_path)
    print('Loading image')
    img_stack = load_image(img_path)
    print('max: ', np.max(img_stack))
    print('min: ', np.min(img_stack))
    print('Loading label')
    lbl_stack = np.int8(load_label(lbl_path))
    print('max: ', np.max(lbl_stack))
    print('min: ', np.min(lbl_stack))
    print('Loading prediction')
    lbl_pred_stack = np.int8(load_label(lbl_pred_path, 1))
    print('max: ', np.max(lbl_pred_stack))
    print('min: ', np.min(lbl_pred_stack))

    print('img type: ', img_stack.dtype)
    print('lbl type: ', lbl_stack.dtype)
    print('lbl_pred type: ', lbl_pred_stack.dtype)

    display(img_stack, lbl_stack, lbl_pred_stack)