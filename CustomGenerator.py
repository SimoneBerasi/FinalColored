from skimage.io import imread
from skimage.transform import resize
import numpy as np
from tensorflow.keras.utils import Sequence
from DataLoader import *
from ColorUtils import prepare_dataset_colorssim
from skimage.color import lab2rgb
from Training import *



'''
    Request as input the list of 
    filenames (pathes) of the training set,
    to read from disk in batches 
    (too big training set for mermory)
'''
class CustomSequence(Sequence):

    def __init__(self, filenames_in, batch_size, color_space = 'cielab', shuffle = True, max = 102., patch_size=128, n_patches=200, color_only=False, grey_only = False):       #label not provided as x = y
        self.max = max
        self.color_space = color_space
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.shuffle = shuffle
        self.x = filenames_in
        self.datalen = len(filenames_in)
        self.indexes = np.arange(self.datalen)
        self.counter=0
        self.color_only = color_only
        self.grey_only = grey_only
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = np.array(self.indexes[index*self.batch_size:(index+1)*self.batch_size])
        filenames_batch = [self.x[i] for i in batch_indexes]
        #filenames_batch = self.x[batch_indexes]

        x_batch = load_patches_from_filenames(filenames_batch, self.patch_size, True, self.n_patches, grayscale=False)
        #visualize_results(x_batch[0], x_batch[1], "a")
        if self.color_space == 'cielab' and self.grey_only == False:
            x_batch = prepare_dataset_colorssim(x_batch)
            x_batch = x_batch / self.max
        else:
            x_batch = x_batch / 255.
        #print("getting an item of shape :")
        #print(x_batch.shape)

        if self.color_only:
            return x_batch[:,:,:, 1:3], x_batch[:,:,:, 1:3]
        if self.grey_only:
            return x_batch[:,:,:, 0], x_batch[:,:,:, 0]

        return x_batch, x_batch

    def __len__(self):
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

'''
class CustomSequence(Sequence):

    def __init__(self, filenames_in, batch_size, color_space = 'cielab', shuffle = True, max = 102., patch_size=128, n_patches=200):       #label not provided as x = y
        self.max = max
        self.color_space = color_space
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.shuffle = shuffle
        self.x = filenames_in
        self.datalen = len(filenames_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = np.array(self.indexes[index*self.batch_size:(index+1)*self.batch_size])
        filenames_batch = [self.x[i] for i in batch_indexes]
        #filenames_batch = self.x[batch_indexes]

        x_batch = load_patches_from_filenames(filenames_batch, self.patch_size, True, self.n_patches, grayscale=False)
        #visualize_results(x_batch[0], x_batch[1], "a")
        if self.color_space == 'cielab':
            x_batch = prepare_dataset_colorssim(x_batch)
            x_batch = x_batch / self.max
        else:
            x_batch = x_batch / 255.
        #print("getting an item of shape :")
        #print(x_batch.shape)


        return x_batch, x_batch

    def __len__(self):
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)
            '''