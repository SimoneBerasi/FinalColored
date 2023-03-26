import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from Utils import visualize_results
import hashlib


def load_patches_from_filenames(filenames, patch_size, random, n_patches=3, grayscale=False):

    # open text file
    text_file = open("names.txt", "a")

    # write string to file
    text_file.write(filenames[0] + '\n')

    # close file
    text_file.close()


    patches=[]
    for file in filenames:
        file_patches, _ = np.array(load_patches_from_file(file, patch_size, random, n_patches, False))
        patches.append(file_patches)
        #visualize_results(file_patches[0], file_patches[0], "a")

    patches = np.array(patches)


    patches = np.reshape(patches, (patches.shape[0]*patches.shape[1], patch_size, patch_size, 3))

    return patches

def load_patches_from_file(file, patch_size, random, n_patches=3, stride=32, grayscale=False):
    if grayscale:
        im1 = cv2.imread(file)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        im1 = cv2.imread(file)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)


    #np.random.seed(hash(file)%99999)


    cropped = []
    if (random == True):
        for _ in range(n_patches):
            j = np.random.randint(0, im1.shape[0] - patch_size)
            i = np.random.randint(0, im1.shape[1] - patch_size)
            cropped.append(im1[j:j + patch_size, i:i + patch_size])
    else:
        for j in range(int((im1.shape[0] - patch_size) / stride) + 1):
            for i in range(int((im1.shape[1] - patch_size) / stride) + 1):
                cropped.append(im1[(j * stride):(j * stride) + patch_size, (i * stride):(i * stride) + patch_size])

    return cropped, im1


def load_patches(folder, patch_size, random=True, n_patches=3, stride=32, grayscale=True):
    patches = []
    for file in os.listdir(folder):
        if file.endswith(".bmp") or file.endswith(".tif") or file.endswith(".png"):
            ret, _ = load_patches_from_file(os.path.join(folder, file), patch_size, random, n_patches, stride, grayscale)
            for r in ret:
                # plt.imshow(r)
                # plt.show()
                patches.append(r)
    return patches


def load_gt_from_file(file):
    im1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return im1 / 255


def load_full_images(folder, grey_scale):
    images = []
    for item in sorted(os.listdir(folder)):
        if item.endswith(".bmp") or item.endswith(".tif") or item.endswith(".png"):
            if grey_scale:
                im1 = cv2.imread(os.path.join(folder, item), cv2.IMREAD_GRAYSCALE)
            else:
                im1 = cv2.imread(os.path.join(folder, item))
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            images.append(im1)
    return images


def load_images(folder, patch_size, random=True, n_patches=3, stride=32, cut_size=None, grayscale=True):
    images = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".bmp") or file.endswith(".tif") or file.endswith(".png"):
            _, img = load_patches_from_file(os.path.join(folder, file), patch_size, random, n_patches, stride, cut_size,grayscale)
            images.append(img)
    return images


def load_patches_from_image(im1, patch_size, random, n_patches=3, stride=32):

    cropped = []
    if (random == True):
        for _ in range(n_patches):
            j = np.random.randint(0, im1.shape[0] - patch_size)
            i = np.random.randint(0, im1.shape[1] - patch_size)
            cropped.append(im1[j:j + patch_size, i:i + patch_size])
    else:
        for j in range(int((im1.shape[0] - patch_size) / stride) + 1):
            for i in range(int((im1.shape[1] - patch_size) / stride) + 1):
                cropped.append(im1[(j * stride):(j * stride) + patch_size, (i * stride):(i * stride) + patch_size])
        cropped = np.array(cropped)
    return cropped, im1