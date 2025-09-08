import os
import glob
import tqdm
import skimage.io as skio
from natsort import natsorted
import numpy as np
from skimage import draw
import json
import skimage.exposure as skex
import time
import skimage as ski
from pathlib import Path

from multiprocessing import Pool, cpu_count


class hipctPreprocessor(object):
    def __init__(self, input_dir, output_dir, convert_to_8_bit=True, masked=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.convert_to_8_bit = convert_to_8_bit
        self.json_dict = {}

        first_im = os.listdir(input_dir)[0]
        self.input_type = first_im.split('.')[-1]
        if self.input_type not in ['jp2', 'tif']:
            raise ValueError(f'Input type {self.input_type} not supported. Please use jp2 or tif.')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.input_type == 'tif':
            first_im = skio.imread(os.path.join(input_dir, first_im), plugin='tifffile')
        else:
            first_im = skio.imread(os.path.join(input_dir, first_im), plugin='imageio')
        
        if masked:
            if first_im.shape[0] != first_im.shape[1]:
                raise ValueError(f'Input image is not square. Please use a square image for masking.')
            self.diameter = first_im.shape[1]
        self.masked = masked

        self.dtype = first_im.dtype

    def _generate_circular_mask(self):
        '''
        This function generates a circular mask for HiP-CT high resolution data.
        Args:
            None
        Returns:
            mask: np.array, circular mask, circle -> 1, outside -> 0
        '''
        rad = self.diameter // 2
        mask = np.zeros((self.diameter, self.diameter), dtype=np.uint8)
        rr, cc = draw.disk(center=(rad, rad), radius=rad)
        mask[rr,cc] = 1
        return mask
    
    def _generate_json_dict(self):
        '''
        This function generates a json dictionary for the normalisation parameters.
        Args:
            None
        Returns:
            json_dict: dict, dictionary of the normalisation parameters
        '''
        self.json_dict['input_type'] = self.input_type
        self.json_dict['convert_to_8_bit'] = self.convert_to_8_bit
        self.json_dict['masked'] = self.masked
        return self.json_dict
    
    def to_tif(self, im_path):
        '''
        This function converts the jp2 image to tif image, with or without 8 bit conversion.
        Args:
            im_path: str, path to the image
        Returns:    
            None
        '''
        if self.input_type == 'tif':
            image = skio.imread(im_path, plugin='tifffile')
        else:
            image = skio.imread(im_path, plugin='imageio')
        if self.convert_to_8_bit and self.dtype != np.uint8:
            # Normalised based on maximum and minimum of each slice
            #image_save = (image - image.min())/(image.max()-image.min())
            image_save = ski.util.img_as_ubyte(image)
        else:
            # Convert to tif without 8 bit conversion
            image_save = image
        if self.masked:
            mask = self._generate_circular_mask()
            image_save = image_save * mask
        tif_file = os.path.join(self.output_dir, os.path.basename(im_path).replace(f'{self.input_type}', 'tif'))
        skio.imsave(tif_file, image_save, plugin='tifffile', check_contrast=False)
    
    def convert_to_tif_multiprocessing(self):
        '''
        This function uses multiprocessor to convert the jp2 image to tif image, with or without 8 bit conversion.
        This function will use _to_tif function to convert the image slice by slice.
        Args:
            None
        Returns:
            None
        '''
        im_files = glob.glob(os.path.join(self.input_dir, f'*.{self.input_type}'))
        file_num = len(im_files)
        im_files = natsorted(im_files)
        print(f'Number of files: {file_num} \n File type: {self.dtype} \n Converting to 8 bit: {self.convert_to_8_bit} \n Masked: {self.masked}')

        self._generate_json_dict()
        json_file_path = os.path.join(self.output_dir, 'to_tif_parameters.json')
        with open(json_file_path, 'w') as f:
            json.dump(self.json_dict, f)
        print(f'Json file saved at: {json_file_path}')
    
        num_cpus = int(cpu_count()/2)
        print(f'Multi-processing with {num_cpus} CPUs...')
        # multi-processing
        with Pool(num_cpus) as p:
            p.map(self.to_tif, im_files)
        print(f'Conversion completed. Tif images saved at: {self.output_dir}')


###############
# Normalisation
###############
class hipctNormaliser(hipctPreprocessor):
    def __init__(self, input_dir, output_dir, convert_to_8_bit=True, masked=True, threshold=0.01):
        super().__init__(input_dir, output_dir, convert_to_8_bit=convert_to_8_bit, masked=masked)
        self.threshold = threshold

    def _normalise(self, im_path):
        '''
        This function normalises the image slice by slice.
        Args:
            im_path: str, path to the image
        Returns:
            norm_slice: np.array, normalised slice
        '''
        # reading image
        if self.input_type == 'tif':
            image = skio.imread(im_path, plugin='tifffile')
        else:
            image = skio.imread(im_path, plugin='imageio')
        # define bins based on image type
        if self.dtype == np.uint8:
            bins = 256
        elif self.dtype == np.uint16:
            bins = 65536
        else:
            raise ValueError(f'Image type {self.dtype} not supported.')
        # calculate the cumulative distribution function and find the lower and upper bounds based on the threshold
        cdf, bin_centers_cdf = skex.cumulative_distribution(image, nbins=bins)
        for i in range(len(cdf)):
            if cdf[i] > self.threshold:
                l_bound = i-1
                break
        for i in range(len(cdf)):
            if cdf[i] > 1-self.threshold:
                u_bound = i-1
                break
        # normalise the image based on the lower and upper bounds
        norm_slice = skex.rescale_intensity(image, in_range=(bin_centers_cdf[l_bound], bin_centers_cdf[u_bound]), out_range=(0, bins-1))
        norm_slice = norm_slice.astype(self.dtype)
        # convert to 8 bit if required
        if self.convert_to_8_bit and self.dtype != np.uint8:
            norm_slice = ski.util.img_as_ubyte(norm_slice)
        # apply mask if required
        if self.masked:
            mask = self._generate_circular_mask()
            norm_slice = norm_slice * mask
        # save the normalised image
        tif_file = os.path.join(self.output_dir, os.path.basename(im_path).replace(f'{self.input_type}', 'tif'))
        skio.imsave(tif_file, norm_slice, plugin='tifffile', check_contrast=False)
        # save the normalisation parameters
        logging = {}
        logging[os.path.basename(im_path)] = {'lower_bound': float(bin_centers_cdf[l_bound]), 
                                              'upper_bound': float(bin_centers_cdf[u_bound])}
        return logging

    def normalise_multiprocessing(self):
        '''
        This function uses multiprocessor to normalise the image slice by slice.
        This function will use _normalise function to normalise the image slice by slice.
        Args:
            None
        Returns:
            None
        '''
        im_files = glob.glob(os.path.join(self.input_dir, f'*.{self.input_type}'))
        file_num = len(im_files)
        im_files = natsorted(im_files)
        print(f'Number of files: {file_num} \n File type: {self.dtype} \n Converting to 8 bit: {self.convert_to_8_bit} \n Masked: {self.masked}')
        
        self._generate_json_dict()

        num_cpus = int(cpu_count()/2)
        print(f'Multi-processing with {num_cpus} CPUs...')
        # multi-processing
        with Pool(num_cpus) as p:
            logging_results = p.map(self._normalise, im_files)
        p.join()
        print(f'Conversion completed. Tif images saved at: {self.output_dir}')

        # combine the logging results
        for log in logging_results:
            for key in log.keys():
                print(f'Key: {key}')
                print(f'Log: {log[key]}')
                self.json_dict[key] = log[key]
        json_file_path = os.path.join(self.output_dir, 'normalisation_parameters.json')
        with open(json_file_path, 'w') as f:
            json.dump(self.json_dict, f, indent=4)
        print(f'Json file saved at: {json_file_path}')


###############
# CLAHE
###############

class hipctClahe(hipctPreprocessor):
    def __init__(self, input_dir, output_dir, convert_to_8_bit=True, masked=True, grid_size=8, clip_limit=0.01):
        super().__init__(input_dir, output_dir, convert_to_8_bit=convert_to_8_bit, masked=masked)
        self.grid_size = grid_size
        self.clip_limit = clip_limit
        # partitions of a 3d volume to be applied CLAHE due to the memory limited

    def _clahe(self, im_path):
        '''
        This function applies CLAHE to the image slice by slice.
        Args:
            im_path: str, path to the image
        Returns:
            None
        '''
        # reading image
        if self.input_type == 'tif':
            image = skio.imread(im_path, plugin='tifffile')
        else:
            image = skio.imread(im_path, plugin='imageio')

        #print(f'Image shape: {image.shape}')


        # define bins based on image type
        if self.dtype == np.uint8:
            bins = 256
        elif self.dtype == np.uint16:
            bins = 65536
        else:
            raise ValueError(f'Image type {self.dtype} not supported.')
        
        # calculate the kernel size based on the grid size
        if image.ndim == 2:
            k_size = (image.shape[0] // self.grid_size, image.shape[1] // self.grid_size)
        elif image.ndim == 3:
            k_size = (image.shape[0] // self.grid_size, image.shape[1] // self.grid_size, image.shape[2] // self.grid_size)
        else:
            raise ValueError(f'Image dimension {image.ndim} not supported.')
        #print(f'Kernel size: {k_size}')

        # apply CLAHE
        clahe_im = skex.equalize_adapthist(image, kernel_size=k_size, clip_limit=self.clip_limit, nbins=bins)
        # convert to 8 bit if required
        if self.convert_to_8_bit and self.dtype != np.uint8:
            clahe_im = ski.util.img_as_ubyte(clahe_im)
        # apply mask if required
        if self.masked:
            mask = self._generate_circular_mask()
            clahe_im = clahe_im * mask
        # save the CLAHE image
        tif_file = os.path.join(self.output_dir, os.path.basename(im_path).replace(f'{self.input_type}', 'tif'))
        skio.imsave(tif_file, clahe_im, plugin='tifffile', check_contrast=False)
        

    def clahe_multiprocessing_2d(self):
        '''
        This function uses multiprocessor to apply CLAHE to the image slice by slice.
        This function will use _clahe function to apply CLAHE to the image slice by slice.
        Args:
            None
        Returns:
            None
        '''
        # read the image files
        im_files = glob.glob(os.path.join(self.input_dir, f'*.{self.input_type}'))
        file_num = len(im_files)
        im_files = natsorted(im_files)
        print(f'Number of files: {file_num} \n File type: {self.dtype} \n Converting to 8 bit: {self.convert_to_8_bit} \n Masked: {self.masked}')

        self._generate_json_dict()

        num_cpus = int(cpu_count()/2)
        print(f'Multi-processing with {num_cpus} CPUs...')
        # multi-processing
        with Pool(num_cpus) as p:
            p.map(self._clahe, im_files)
        p.join()
        print(f'Conversion completed. Tif images saved at: {self.output_dir}')

        # combine the logging results
        self.json_dict['grid_size'] = float(self.grid_size)
        self.json_dict['clip_limit'] = float(self.clip_limit)     
        json_file_path = os.path.join(self.output_dir, 'clahe_2d_parameters.json')
        with open(json_file_path, 'w') as f:
            json.dump(self.json_dict, f)
        print(f'Json file saved at: {json_file_path}') 
    
    def clahe_3d(self):
        '''
        This function applies CLAHE to the image volume.
        Args:
            None
        Returns:
            None
        '''
        # read the volume
        im_files = glob.glob(os.path.join(self.input_dir, f'*.{self.input_type}'))
        file_num = len(im_files)
        im_files = natsorted(im_files)
        print(f'Number of files: {file_num} \n File type: {self.dtype} \n Converting to 8 bit: {self.convert_to_8_bit} \n Masked: {self.masked}')

        self._generate_json_dict()

        for im_path in tqdm.tqdm(im_files):
            self._clahe(im_path)

        self.json_dict['grid_size'] = float(self.grid_size)
        self.json_dict['clip_limit'] = float(self.clip_limit)
        json_file_path = os.path.join(self.output_dir, 'clahe_3d_parameters.json')
        with open(json_file_path, 'w') as f:
            json.dump(self.json_dict, f)
        print(f'Json file saved at: {json_file_path}')

##########
# Convert
##########

def generate_training_patches(image_dir, label_dir, save_dir, split_size):
    '''
    Split into 128x128x128 patches for training.
    Args:
        image_dir: str, path to the image to be split
        label_dir: str, path to the label to be split
        save_train_dir: str, path to the directory to save the training patches
        split_size: tuple, size of the patches
    '''
    # Convert to Path objects for easier handling
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    save_dir = Path(save_dir)

    # Validate required directories
    for path in [image_dir, label_dir]:
        if not path.exists():
            raise ValueError(f"{path} does not exist.")

    # Create necessary subdirectories
    for subdir in ["", "cubes", "labels"]:
        (save_dir / subdir).mkdir(parents=True, exist_ok=True)

    # if split-_size is not a tuple
    if isinstance(split_size, int):
        split_size = (split_size, split_size, split_size)
    elif isinstance(split_size, list):
        split_size = tuple(split_size)
    elif not isinstance(split_size, tuple):
        raise ValueError(f'Split size {split_size} is not a valid type. Please use int, list or tuple.')
    if len(split_size) != 3:
        raise ValueError(f'Split size {split_size} is not a valid size. Please use a tuple of 3 integers.')

    cube_files = glob.glob(os.path.join(image_dir, '*.tif'))
    cube_files = natsorted(cube_files)
    print('Number of cube files: ', len(cube_files))
    pbar = tqdm.tqdm(cube_files)
    for cube_file in pbar:
        pbar.set_description('Processing cube file: %s' % cube_file.split('/')[-1])
        idx = os.path.basename(cube_file).split('.')[0].split('_')[-1]
        cube = skio.imread(cube_file, plugin='tifffile')
        v, h, w = np.shape(cube)
        num = 0
        for i in range(0, v, split_size[0]):
            for j in range(0, h, split_size[1]):
                for k in range(0, w,  split_size[2]):
                    patch = cube[i:i+split_size[0], j:j+split_size[1], k:k+split_size[2]]
                    skio.imsave(os.path.join(save_dir,'cubes', 'cube'+str(idx)+'_'+str(num)+'_T0.tif'), patch, check_contrast=False)
                    num += 1
    print('Completed splitting the data patches!')

    label_files = glob.glob(os.path.join(label_dir, '*.tif'))
    label_files = natsorted(label_files)
    print('Number of label files: ', len(label_files))
    pbar = tqdm.tqdm(label_files)
    for label_file in pbar:
        pbar.set_description('Processing label file: %s' % label_file.split('/')[-1])
        idx = os.path.basename(label_file).split('.')[0].split('_')[-1]
        label = skio.imread(label_file, plugin='tifffile')
        v, h, w = np.shape(label)
        num = 0
        for i in range(0, v, split_size[0]):
            for j in range(0, h, split_size[1]):
                for k in range(0, w, split_size[2]):
                    patch = label[i:i+split_size[0], j:j+split_size[1], k:k+split_size[2]]
                    skio.imsave(os.path.join(save_dir,'labels', 'label'+str(idx)+'_'+str(num)+'_T0.tif'), patch, check_contrast=False)
                    num += 1
    print('Completed splitting the label patches!')

def convert_volume_to_slices(volume_path, save_path, save_start_idx=0):
    '''
    The prediction from NN is a column. The function is to seperate the column into slices, matching to each single tif image.
    Args:
        pred_path: str, path to the prediction file
        save_path: str, path to save the slices 
    '''
    vol = skio.imread(volume_path, plugin='tifffile')
    print(f'Prediction shape: {vol.shape}')
    vol_slice_len = len(vol)
    print(f'Number of slices: {vol_slice_len}')

    for i in tqdm.tqdm(range(vol_slice_len)):
        save_idx = i + save_start_idx
        slice_save_path = os.path.join(save_path, f'slice_{save_idx}.tif')
        skio.imsave(slice_save_path, vol[i], plugin='tifffile', check_contrast=False)
    print(f'Slices saved at: {save_path}')
    print('Conversion completed.')

def convert_slices_to_volume(slice_dir, save_dir, im_type='tif', partition=1, overlap=0):
    '''
    This function converts the slices to a volume.
    Args:
        slice_dir: str, path to the slice directory
        save_path: str, path to save the volume
    '''
    slice_files = glob.glob(os.path.join(slice_dir, f'*.{im_type}'))
    slice_files = natsorted(slice_files)
    print(f'Number of slices: {len(slice_files)}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    num_slices = len(slice_files)
    new_num_slices = num_slices + overlap * (partition - 1)
    partition_size = new_num_slices // partition

    start = 0
    end = 0
    for i in range(partition):
        start = end - overlap
        if start < 0:
            start = 0
        end = start + partition_size
        if end > num_slices:
            end = num_slices
        if i == partition - 1:
            end = num_slices
        
        if i == 0 or i == 1 or i == 2:
            continue
        print(f'Processing partition {i+1}/{partition}...')
        slice_files_partition = slice_files[start:end]
        print(f'Partition start: {start}, end: {end-1}')
        print(f'Number of slices in partition: {len(slice_files_partition)}')
        volume = []
        for slice_file in tqdm.tqdm(slice_files_partition):
           if im_type == 'tif':
               slice = skio.imread(slice_file, plugin='tifffile')
           else:
               slice = skio.imread(slice_file, plugin='imageio')
           volume.append(slice)
        volume = np.asarray(volume)
        print(f'Slices shape: {volume.shape}')
        save_path = os.path.join(save_dir, f'volume_s{start}_s{end-1}.tif')
        skio.imsave(save_path, volume, plugin='tifffile', check_contrast=False)
        print(f'Volume saved at: {save_dir}')
        print('Conversion completed.')

def devide_subvolumes(big_vol_path, segments=2, overlap_depth_in_pixels=0):
    '''
    This function divides a big volume into subvolumes.
    Args:
        big_vol_path: str, path to the big volume
        segments: int, number of segments to divide the volume into
        overlap_depth_in_pixels: int, overlap depth in pixels
    Returns:
        None
    '''
    big_vol = skio.imread(big_vol_path, plugin='tifffile')
    print(f'Big volume shape: {big_vol.shape}')
    vol_depth = big_vol.shape[0]

    segment_size = vol_depth // segments
    print(f'Segment size: {segment_size}')

    if overlap_depth_in_pixels >= segment_size:
        raise ValueError(f'Overlap depth {overlap_depth_in_pixels} is greater than or equal to segment size {segment_size}. Please reduce the overlap depth.')
    
    start = 0
    end = 0
    for i in tqdm.tqdm(range(segments)):
        start = end - overlap_depth_in_pixels if i > 0 else 0
        end = start + segment_size if i < segments - 1 else vol_depth-1
        
        if start < 0:
            start = 0
        if end > vol_depth:
            end = vol_depth
        
        subvol = big_vol[start:end]
        save_path = os.path.join(os.path.dirname(big_vol_path), os.path.basename(big_vol_path).replace('.tif', f'_subvol_{start}_{end}.tif'))
        skio.imsave(save_path, subvol, plugin='tifffile', check_contrast=False)
        print(f'Subvolume saved at: {save_path}')
        print(f'Subvolume shape: {subvol.shape}')
    print('Subvolume division completed.')