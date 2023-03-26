import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from CustomGenerator import CustomSequence

from Steerables.SCFpyr_TF import SCFpyr_TF
import Steerables.utils as utils
from DataLoader import load_patches
import matplotlib.pyplot as plt
from Steerables.metrics_TF import Metric_win
from ColorUtils import OCCD, build_codebook, compute_all_color_distances, simple_color_metric, prepare_dataset_colorssim

from Utils import preprocess_data, visualize_results
from NNModels import *
MAX_WOOD = 102


import argparse
import configparser
import os

from NNModels import Model_noise_skip_bigger

training_dataset_path = '/home/simo/Desktop/Thesis Projects/AnomalyDetectionBionda/Dataset/MVTec_Data/wood/train/good'

loss_type = 'color_cwssim_loss'
window_size = 7
scales = 5
orients = 5



lr = 1e-3
decay_fac = 0.3
decay_step = 8
epoch = 50
batch_size = 32
patch_size = 128
save_period = 5
num_channel = 3
pad_size = 3
color_parameter = 40
color_space = 'cielab'
steps_per_epoch = 250
generator_batch_size = 1
n_patches = 250
image_size = 1024

def scheduler(epoch):
    return lr * decay_fac ** (np.floor(epoch / decay_step))



#takes images as rgb or lab. computes cwssim on first channel
def cwssim_loss_3channel(y_true, y_pred):
    return cwssim_loss(tf.expand_dims(y_true[:, :, :, 0], 3), tf.expand_dims(y_pred[:, :, :, 0], 3))



#heavy on resources, used for ecperiments
def perceptual_loss(y_true, y_pred):
    y_true = tf.keras.applications.vgg16.preprocess_input(y_true)
    y_pred = tf.keras.applications.vgg16.preprocess_input(y_pred)
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(patch_size, patch_size, 3))
    loss_model = tf.keras.Model(inputs=vgg.input,
    outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


#Definitive metric, requires CIELAB as input
def color_cwssim_loss(y_true, y_pred):

    paddings_cwssim = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size]])

    y_true_cwssim = tf.pad(y_true[:, :, :, 0], paddings_cwssim, "SYMMETRIC")
    y_pred_cwssim = tf.pad(y_pred[:, :, :, 0], paddings_cwssim, "SYMMETRIC")

    metric_tf = Metric_win (patch_size, window_size=window_size, pad_size = 6)
    cwssim_scores = metric_tf.CWSSIM(tf.expand_dims(y_pred_cwssim, 3), tf.expand_dims(y_true_cwssim, 3), height=scales, orientations=orients)

    #Here the computation of color factor, only on a and b planes
    color_loss = tf.keras.losses.MSE(y_true[:,:,:, 1:3], y_pred[:,:,:, 1:3])

    #TODO try with + or imcrease color_parameter for a faster converging. Or use 2 different aes
    color_scores = tf.math.reduce_mean(1. - color_loss/45)
    loss = tf.math.reduce_mean(1. - cwssim_scores*color_scores)
    return loss

def color_cwssim_loss_no_padding(y_true, y_pred):


    metric_tf = Metric_win (patch_size, window_size=window_size, pad_size = 0)
    cwssim_scores = metric_tf.CWSSIM(tf.expand_dims(y_pred[:,:,:,0],3) , tf.expand_dims(y_true[:,:,:,0],3), height=scales, orientations=orients)

    #Here the computation of color factor, only on a and b planes
    color_loss = tf.keras.losses.MSE(y_true[:,:,:, 1:3], y_pred[:,:,:, 1:3])

    #TODO try with + or imcrease color_parameter for a faster converging. Or use 2 different aes
    color_scores = tf.math.reduce_mean(1. - color_loss/color_parameter)
    loss = tf.math.reduce_mean(1. - cwssim_scores*color_scores)
    return loss

def cwssim_only_metric(y_true, y_pred):
    paddings_cwssim = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size]])

    y_true_cwssim = tf.pad(y_true[:, :, :, 0], paddings_cwssim, "SYMMETRIC")
    y_pred_cwssim = tf.pad(y_pred[:, :, :, 0], paddings_cwssim, "SYMMETRIC")

    metric_tf = Metric_win (patch_size, window_size=window_size, pad_size = 6)
    cwssim_scores = metric_tf.CWSSIM(tf.expand_dims(y_pred_cwssim, 3), tf.expand_dims(y_true_cwssim, 3), height=scales, orientations=orients)


    loss = tf.math.reduce_mean(1. - cwssim_scores)
    return loss

def color_only_metric(y_true, y_pred):
    color_loss = tf.keras.losses.MSE(y_true[:,:,:, 1:3], y_pred[:,:,:, 1:3])


    color_scores = tf.math.reduce_mean(1. - color_loss/color_parameter)
    loss = tf.math.reduce_mean(1. - color_scores)
    return loss

def color_cwssim_loss_SUM(y_true, y_pred):

    paddings_cwssim = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size]])

    y_true_cwssim = tf.pad(y_true[:, :, :, 0], paddings_cwssim, "SYMMETRIC")
    y_pred_cwssim = tf.pad(y_pred[:, :, :, 0], paddings_cwssim, "SYMMETRIC")

    metric_tf = Metric_win (patch_size, window_size=window_size, pad_size = 6)
    cwssim_scores = metric_tf.CWSSIM(tf.expand_dims(y_pred_cwssim, 3), tf.expand_dims(y_true_cwssim, 3), height=scales, orientations=orients)

    #Here the computation of color factor, only on a and b planes
    color_loss = tf.keras.losses.MSE(y_true[:,:,:, 1:3], y_pred[:,:,:, 1:3])

    #TODO try with + or imcrease color_parameter for a faster converging. Or use 2 different aes
    color_loss = tf.math.reduce_mean(color_loss)
    loss = tf.math.reduce_mean(1. - cwssim_scores)

    return loss + color_loss


def cwssim_occd_loss(occd_model):
    def cwssim_occd_internal_loss(y_true, y_pred):
        batch_number = y_true.shape[0]

        paddings_cwssim = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size]])
        paddings_color = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])

        y_true_cwssim = tf.pad(y_true[:, :, :, 0], paddings_cwssim, "SYMMETRIC")
        y_pred_cwssim = tf.pad(y_pred[:, :, :, 0], paddings_cwssim, "SYMMETRIC")

        metric_tf = Metric_win(patch_size, window_size=window_size)
        cwssim_scores = metric_tf.CWSSIM(tf.expand_dims(y_true_cwssim, 3), tf.expand_dims(y_pred_cwssim, 3),
                                           height=scales, orientations=orients)
        # print(stsim_scores_tf.shape)

        y_true_color = tf.pad(y_true, paddings_color, "SYMMETRIC")
        y_pred_color = tf.pad(y_pred, paddings_color, "SYMMETRIC")

        true_patches = tf.image.extract_patches(images=y_true_color,
                                           sizes=[1, 7, 7, 1],
                                           strides=[1, 3, 3, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')

        pred_patches = tf.image.extract_patches(images=y_pred_color,
                                           sizes=[1, 7, 7, 1],
                                           strides=[1, 3, 3, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')

        #TODO remove hard-coding
        true_patches = tf.reshape(true_patches, (batch_number, 43*43, window_size, window_size, 3))
        pred_patches = tf.reshape(pred_patches, (batch_number, 43*43, window_size, window_size, 3))

        x = tf.stack((true_patches, pred_patches))
        internal_batch = true_patches.shape[0]
        x = tf.reshape(x, (internal_batch*43*43, 2, window_size, window_size, 3))

        occd_loss = occd_model(x)
        occd_loss = tf.reduce_mean(0.+occd_loss)
        cwssim_scores = tf.reduce_mean(cwssim_scores)
        cwssimc = (1. - cwssim_scores*(1-occd_loss))

        return cwssimc
    return cwssim_occd_internal_loss




def cwssim_loss(y_true, y_pred):
    metric_tf = Metric_win (patch_size, window_size=window_size)
    cwssim_scores = metric_tf.CWSSIM(y_pred, y_true, height=scales, orientations=orients)
    loss = tf.math.reduce_mean(1. - cwssim_scores)
    return loss


#CW-SSIM averaged on rgb channels
def cwssim_3channel_loss(y_true, y_pred):

    r_cwssim = cwssim_loss(tf.expand_dims(y_true[:, :, :, 0], 3), tf.expand_dims(y_pred[:, :, :, 0], 3))
    g_cwssim = cwssim_loss(tf.expand_dims(y_true[:, :, :, 1], 3), tf.expand_dims(y_pred[:, :, :, 1], 3))
    b_cwssim = cwssim_loss(tf.expand_dims(y_true[:, :, :, 2], 3), tf.expand_dims(y_pred[:, :, :, 2], 3))

    #return tf.reduce_max([r_cwssim, g_cwssim, b_cwssim])
    return tf.reduce_mean([r_cwssim, g_cwssim, b_cwssim])

def ssim_loss (y_true, y_pred):
    return tf.reduce_mean (1. - tf.image.ssim(y_true, y_pred, 1.0))

def ms_ssim_loss (y_true, y_pred):
    return tf.reduce_mean (1. - tf.image.ssim_multiscale(y_true, y_pred, 1.0))

def psnr_color_loss(y_true, y_pred):
    mse = tf.keras.losses.MSE(y_true, y_pred)
    #psnr = 10 * log10(1 ./ mean(mse,3));
    #print(mse)
    psnr = tf.experimental.numpy.log10(1 / tf.math.sqrt(mse + 1e-2))
    return 1 - psnr

def l2_loss (y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def train():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #tf.config.run_functions_eagerly(True) #TODO try to remove this
    tf.keras.backend.set_floatx('float64')


    filenames = os.listdir(training_dataset_path)

    path_filnames = []
    for name in filenames:
        path_filnames.append(training_dataset_path + '/' + name)

    #todo rstore
    training_batch_generator = CustomSequence(path_filnames, 1, color_space=color_space, max=102, patch_size = patch_size, n_patches=100 )
    #training_batch_generator = CustomSequence(path_filnames, 1, color_space=color_space, max=102, patch_size = patch_size, n_patches=25, color_only=True )


    #train_dataset = tf.data.Dataset.from_generator(lambda: training_batch_generator, output_types=(tf.float64, tf.float64))
    #train_dataset = train_dataset.repeat(1000000)



    loss_function = None
    for loss in [cwssim_loss, color_cwssim_loss_SUM, ssim_loss, ms_ssim_loss, l2_loss, cwssim_occd_loss, perceptual_loss,psnr_color_loss, color_cwssim_loss, cwssim_loss_3channel]:
        if (loss.__name__ == loss_type):
            loss_function = loss

    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('Weights','new_weights','check_epoch{epoch:02d}.h5'), save_weights_only=True, period=save_period))

    autoencoder = Model_noise_skip_bigger_old(input_shape=(None, None, 3))

    #todo remove
    autoencoder.load_weights('Weights/new_weights/carpet150.h5')

    if loss_function.__name__ == 'cwssim_occd_loss':
        occd_model = keras.models.load_model('OCCD_model_50k')
        occd_model.summary()

    #autoencoder.compile(optimizer='adam', loss=loss_function, metrics=[color_only_metric])
    #autoencoder.compile(optimizer='adam', loss=color_cwssim_loss_SUM, metrics=[color_only_metric, cwssim_only_metric])
    autoencoder.compile(optimizer='adam', loss=color_cwssim_loss, metrics=[color_only_metric, cwssim_only_metric])


    #ideal
    #autoencoder.fit(training_batch_generator,steps_per_epoch=250, use_multiprocessing=False, workers=8, epochs=epoch, shuffle=True,  callbacks=callbacks, initial_epoch = 110)

    autoencoder.fit(training_batch_generator,steps_per_epoch=500, verbose=1, use_multiprocessing=False, workers=8, epochs=epoch, shuffle=True,  callbacks=callbacks, initial_epoch = 10)




























