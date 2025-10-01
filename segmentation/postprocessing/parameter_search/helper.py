import tqdm
from natsort import natsorted
import numpy as np
from scipy.stats import qmc
from skimage.morphology import closing, cube, label
from napari_simpleitk_image_processing import label_statistics
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import pandas as pd
import os
from glob import glob
import skimage.io as skio
import csv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


def LHC(sample_size, dim, l_bounds, u_bounds, seed=0):
    '''
    Args:
        sample_size: number of samples (iterations)
        dim: dimension
        l_bounds: lower bounds
        u_bounds: upper bounds
        seed: random seed
    Returns:
        sample_scaled: sampled hyper-parameters
    '''
    if len(l_bounds) != dim or len(u_bounds) != dim:
        raise ValueError('Bounds must have the same dimension as the number of samples')
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n=sample_size)
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    #print('Sampling the hyper-parameters, sample shape: ', sample_scaled.shape)
    return sample_scaled

def get_minimum_vol(resolution=2.58, radius=62.035):
    '''
    Args:
        resolution: pixel spacing of the data, in um
        radius: radius of the glomeruli in um
    Returns:
        minimum size of the glomeruli in pixels
    '''
    vol_in_pixel = (4/3) * np.pi * (radius/resolution)**3
    return vol_in_pixel

def get_maximum_vol(resolution=2.58, radius=112.725):
    '''
    Args:
        resolution: pixel spacing of the data, in um
        radius: radius of the glomeruli in um
    Returns:
        maximum size of the glomeruli in pixels
    '''
    vol_in_pixel = (4/3) * np.pi * (radius/resolution)**3
    return vol_in_pixel

def label_closing(label, cube_size=5):
    '''
    Closing operation on label
    Args:
        label: label to be closed, [512, 512, 512]
        cube_size: size of cube for closing operation, defalt 5 (pixel)
    '''
    label = np.int8(label)
    label = closing(label, cube(cube_size))
    return label

def remove_based_on_size(instance, size):
    '''
    Args:
        instance: a row of the properties table
        size: size threshold
    Returns:
        flag: True if the instance is to be removed, False otherwise
    '''
    flag = False
    if instance['number_of_pixels'] < size and instance['number_of_pixels_on_border'] == 0: # 25000
        flag = True
    return flag

def bounds_calculation(props_table, property_dict):
    bounds = {}
    for i, prop in enumerate(property_dict['property']):
        print(f'Calculating bounds for {prop}...')
        prop_name = prop.split('_')[0]
        prop_values = props_table[prop_name]
        prop_values = natsorted(prop_values)
        print(f'Number of {prop}: {len(prop_values)}')
        print(f'Minimum {prop}: {prop_values[0]}, Maximum {prop}: {prop_values[-1]}')
        lower_bound = prop_values[int(len(prop_values)*property_dict['threshold'][i][0])]
        upper_bound = prop_values[int(len(prop_values)*property_dict['threshold'][i][1])-1]
        print(f'Lower bound: {lower_bound}, Upper bound: {upper_bound}\n')
        bounds[prop] = [lower_bound, upper_bound]
    return bounds

def generate_lhc_sample(bounds, property_dict, save_dir, seed=0, sample_size=20):
    '''
    Args:
        bounds: dictionary of bounds for each property
        property_dict: dictionary of properties and their thresholds
        seed: random seed
        sample_size: number of samples (iterations)
    Returns:
        sample_matrix: sampled hyper-parameters
    '''
    if os.path.exists(os.path.join(save_dir, 'sample_matrix.csv')):
        print('The sample matrix exists, using the existing one...')
        sample_matrix = csv.reader(open(os.path.join(save_dir, 'sample_matrix.csv')))
        sample_matrix = np.array([[float(i) for i in row] for row in sample_matrix])

    else:    
        l_bounds = []
        u_bounds = []
        for prop in property_dict['property']: 
            l_bounds.append(bounds[prop][0])
            u_bounds.append(bounds[prop][1])
        sample_matrix = LHC(sample_size=20, dim=len(property_dict['property']), l_bounds=l_bounds, u_bounds=u_bounds)
        print('Sample_matrix shape: ', sample_matrix.shape)
        print('Finished the hyper-parameters sampling\n')
        with open(os.path.join(save_dir, 'sample_matrix.csv'), 'w') as f:
            for row in sample_matrix:
                f.write(','.join([str(r) for r in row]) + '\n')
    return sample_matrix

def generate_props_table(im, pred, shape=False, perimeter=True, position=True, moments=False):
    '''
    Args:
        im: input images
        pred: binary prediction
    Returns:
        pred_props_table: properties table of the detected objects
    '''
    print('Generating prediction properties...')
    lbl_pred = label(pred)
    pred_props_table = label_statistics(im, lbl_pred, 
                                        shape=shape, 
                                        perimeter=perimeter, 
                                        position=position, 
                                        moments=moments)
    print(f'There are {len(pred_props_table)} objects detected in the label')
    return pred_props_table

def remove_labels(lbl_remove, pred_props_table, lbl_pred):
    '''
    Args:
        lbl_remove: list of labels to be removed
        pred_props_table: properties table of the prediction
        lbl_pred: labeled prediction
    '''
    print('Removing labels...')
    filtered_pred = lbl_pred.copy()
    if len(lbl_remove) == 0:
        pass
    else:
        pbar= tqdm(total=len(lbl_remove))
        for l in lbl_remove:
            pbar.set_description(f'Processing:')
            row = pred_props_table[pred_props_table['label']==l]
            x_start = row['bbox_0'].values[0]
            y_start = row['bbox_1'].values[0]
            z_start = row['bbox_2'].values[0]
            x_range = row['bbox_3'].values[0]
            y_range = row['bbox_4'].values[0]
            z_range = row['bbox_5'].values[0]
            extracted_cube = filtered_pred[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range].copy()
            extracted_cube[extracted_cube == l] = 0
            filtered_pred[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range] = extracted_cube
            pbar.update(1)
        del pbar
    #lbl_pred[lbl_pred > 0] = 1
    #filtered_pred = lbl_pred.astype(np.int8)
    return filtered_pred

def update_props_table(pred_removed, pred_props_table):
    print('Update new prediction properties...')
    for idx in pred_removed:
        if idx in pred_props_table['label'].values:
            pred_props_table = pred_props_table[pred_props_table['label'] != idx]
    return pred_props_table

def reading_data(im_path):
    if im_path.endswith('.tif'):
        im = skio.imread(im_path, plugin='tifffile')
    else:
        print('Data is a folder, loading all the slices...')
        first_slice_path = os.listdir(im_path)[0]
        file_suffix = os.path.basename(first_slice_path).split('.')[-1]
        if file_suffix == 'tif':
            im_slices = natsorted(glob(os.path.join(im_path, '*.tif')))
            im = np.stack([skio.imread(s, plugin='tifffile') for s in im_slices], axis=0)
        elif file_suffix == 'jp2':
            im_slices = natsorted(glob(os.path.join(im_path, '*.jp2')))
            im = np.stack([skio.imread(s) for s in im_slices], axis=0)
        else:
            raise ValueError('Unsupported file format, only tif and jp2 are supported.')
    return im

def check_size_filtered_table(im_path, raw_pred_path, resolution):
    '''
    Args:
        im_path: path to the image stack (tif) or folder of slices (tif, jp2)
        raw_pred_path: path to the raw prediction, directly from nnUNet (tif)
    Returns:
        pred_props_table: properties table after size filtering
    '''

    if os.path.exists(os.path.join(os.path.dirname(raw_pred_path), 'props_table_itk_after_size.csv')):
        print('The properties table exists, using the existing one...')
        pred_props_table = pd.read_csv(os.path.join(os.path.dirname(raw_pred_path), 'props_table_itk_after_size.csv'))
    else:
        print('Reading data...')
        im = reading_data(im_path)

        # read the raw prediction
        raw_pred = reading_data(raw_pred_path)

        im = im[:-1, :, :]
        raw_pred = raw_pred[:-1, :, :]

        print(f'Image shape: {im.shape}')
        print(f'Raw prediction shape: {raw_pred.shape}\n')
        assert im.shape == raw_pred.shape, 'Image and prediction must have the same shape'

        # generate label properties
        print('Generating prediction properties...')
        lbl_pred = label(raw_pred)
        pred_props_table = label_statistics(im, lbl_pred, shape=True, perimeter=True, position=True, moments=False)

        print(f'There are {len(pred_props_table)} objects detected in the label')
        # first remove based on size
        size = get_minimum_vol(resolution=resolution, radius=62.035) # 62.035 um is the minimum radius of glomeruli
        pred_removed = []
        print('Start removing outliers...')
        for index, row in pred_props_table.iterrows():
            if remove_based_on_size(row, size):
                pred_removed.append(int(row['label']))

        removed_based_on_size = len(pred_removed)
        print('Removing ratio: \n', removed_based_on_size/len(pred_props_table))

        pred_after_size_filtered = remove_labels(pred_removed, pred_props_table, lbl_pred)
        pred_after_size_filtered[pred_after_size_filtered > 0] = 1
        pred_after_size_filtered = pred_after_size_filtered.astype(np.int8)

        save_dir = os.path.dirname(raw_pred_path)
        skio.imsave(os.path.join(save_dir, 'pred_after_size_filtered.tif'), pred_after_size_filtered, check_contrast=False)
        print('Finished the size filtered\n')
        del lbl_pred

        # update the properties table
        pred_props_table = update_props_table(pred_removed, pred_props_table)
        print(f'There are {len(pred_props_table)} objects detected in the label')
        # save the props_table_itk
        pred_props_table.to_csv(os.path.join(save_dir, 'props_table_itk_after_size.csv'))
        print('Properties table saved successfully.')
    return pred_props_table

def watershed_big_label(image, size_filtered):
    '''
    Args:
        image: binary image of a single connected component
        size_filtered: the minimum size threshold
    Returns:
        count: number of objects after watershed
    '''
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((2, 2, 2)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    labels = watershed(-distance, markers, mask=image)
    props_table = label_statistics(image, labels, shape=False, perimeter=False, position=False, moments=False)
    count=0
    for index, row in props_table.iterrows():
        if row['number_of_pixels'] > 0.75 * size_filtered:
            count += 1
    return count