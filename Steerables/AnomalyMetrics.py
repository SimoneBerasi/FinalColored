import numpy as np
from skimage.metrics import structural_similarity as ssim
from Steerables.metrics_TF import Metric_win
from Utils import visualize_results, visualize_result3
import tensorflow as tf


pad_size_color = 2
color_win_size = 5 #todo restore 3
img_size = 1024


#Expects image in CIELAB representation
def color_part_metric(x_valid, y_valid, window_size):

    x_valid = tf.reshape(x_valid, (1, x_valid.shape[0], x_valid.shape[1], 3))
    y_valid = tf.reshape(y_valid, (1, y_valid.shape[0], y_valid.shape[1], 3))


    pad_size = int((window_size-1)/2)
    paddings = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])

    x_valid = tf.pad(x_valid, paddings, "SYMMETRIC")
    y_valid = tf.pad(y_valid, paddings, "SYMMETRIC")

    c = tf.keras.losses.MSE(x_valid, y_valid)


    c = tf.reshape(c, (1, x_valid.shape[1],x_valid.shape[2],  1))



    window =(tf.constant(np.ones((window_size,window_size,1,1))/(window_size*window_size))) # average filter, 5x5 works great
    closs = tf.nn.conv2d(c, window, strides=[1, 1, 1, 1], padding="VALID")

    return closs


#Expects input as CIELAB
def cw_ssimcolor_metric(x_valid, y_valid, pad_size_x, pad_size_y, gray_scale, pad_size = 0):
    # Residual map ssim
    ssim_configs = [17, 15, 13, 11, 9, 7, 5, 3]
    '''
    residual_ssim = np.zeros_like(x_valid[:,:,0])
    print(y_valid.shape)
    for win_size in ssim_configs:
        for i in range(0, 3):
            residual_ssim += (1 - ssim(x_valid[:,:,i], y_valid[:,:,i], win_size=win_size, full=True, data_range=1.)[1])
    residual_ssim = residual_ssim / len(ssim_configs)
    residual_ssim = residual_ssim[pad_size_x: residual_ssim.shape[0]-pad_size_x, pad_size_y:residual_ssim.shape[1]-pad_size_y]
    print(y_valid)
    #visualize_results(residual_ssim, y_valid, "ssim")
    '''
    # Residual map cwssim
    paddings_color = tf.constant([ [pad_size_color, pad_size_color], [pad_size_color, pad_size_color], [0, 0]])

    cwssim_configs = [9, 8, 7]
    residual_cwssim = np.expand_dims(np.expand_dims(np.zeros_like(x_valid[:,:,0]), 0), 3) / 1.
    #TODO here check patch_size
    metric_tf_7 = Metric_win(window_size=7, patch_size=1024, pad_size=pad_size)
    for height in cwssim_configs:
        # for i in range(0, 3):
        # residual_cwssim += (1 - metric_tf_7.CWSSIM(np.expand_dims(np.expand_dims(x_valid[:,:,i], 0), 3), np.expand_dims(np.expand_dims(y_valid[:,:,i], 0), 3),
        # height=height, orientations=6, full=True).numpy()[0])

        #print(np.expand_dims(np.expand_dims(x_valid[:,:,0], 0), 3).shape)
        residual_cwssim += metric_tf_7.CWSSIM(np.expand_dims(np.expand_dims(x_valid[:,:,0], 0), 3),
                                                     np.expand_dims(np.expand_dims(y_valid[:,:,0], 0), 3),
                                                     height=height, orientations=6, full=True).numpy()[0]


    residual_cwssim = residual_cwssim / len(cwssim_configs)

    x_valid_color = tf.pad(x_valid, paddings_color, "SYMMETRIC")
    y_valid_color = tf.pad(y_valid, paddings_color, "SYMMETRIC")
    #TODO restore
    c = tf.keras.losses.MSE(x_valid_color[:,:,1:3], y_valid_color[:,:,1:3])
    #c = tf.keras.losses.MSE(x_valid_color, y_valid_color)
    window = (tf.constant(np.ones((color_win_size, color_win_size, 1, 1)) / (color_win_size * color_win_size)))


    c = tf.reshape(c, (1, img_size+pad_size_color*2, img_size+pad_size_color*2))  #this for 1024x1024 TOdo
    #c = tf.reshape(c, (1, 842, 842))  #TODO
    residual_color = metric_tf_7.conv(c, window, False)
    #residual_color = residual_color / tf.reduce_max(residual_color)

    #residual_color = residual_color*10




    #residual_color = (residual_color / np.max(residual_color)) * np.max(residual_cwssim)

    #todo restore
    #residual_color = 1 - residual_color*15
    residual_color = 1 - residual_color*10

    residual_color = tf.squeeze(residual_color)
    residual_cwssim = tf.squeeze(residual_cwssim)





    return residual_color, 1 - residual_cwssim, 1 - residual_color*residual_cwssim





def cw_ssim_metric (x_valid, y_valid, pad_size_x, pad_size_y, gray_scale):
    #Residual map ssim
    ssim_configs = [17, 15, 13, 11, 9, 7, 5, 3]
    '''
    residual_ssim = np.zeros_like(x_valid[:,:,0])
    print(y_valid.shape)
    for win_size in ssim_configs:
        for i in range(0, 3):
            residual_ssim += (1 - ssim(x_valid[:,:,i], y_valid[:,:,i], win_size=win_size, full=True, data_range=1.)[1])
    residual_ssim = residual_ssim / len(ssim_configs)
    residual_ssim = residual_ssim[pad_size_x: residual_ssim.shape[0]-pad_size_x, pad_size_y:residual_ssim.shape[1]-pad_size_y]
    print(y_valid)
    #visualize_results(residual_ssim, y_valid, "ssim")
    '''
    #Residual map cwssim
    cwssim_configs = [9, 8, 7]
    residual_cwssim = np.expand_dims(np.expand_dims(np.zeros_like(x_valid), 0), 3) /1.
    metric_tf_7 = Metric_win (window_size=7, patch_size=1024)
    for height in cwssim_configs:
        #for i in range(0, 3):
            #residual_cwssim += (1 - metric_tf_7.CWSSIM(np.expand_dims(np.expand_dims(x_valid[:,:,i], 0), 3), np.expand_dims(np.expand_dims(y_valid[:,:,i], 0), 3),
                        #height=height, orientations=6, full=True).numpy()[0])

        residual_cwssim += (1.0 - metric_tf_7.CWSSIM(np.expand_dims(np.expand_dims(x_valid, 0), 3),
                                                   np.expand_dims(np.expand_dims(y_valid, 0), 3),
                                                   height=height, orientations=6, full=True).numpy()[0])

    residual_cwssim = residual_cwssim/len(cwssim_configs)
    residual_cwssim = np.squeeze(residual_cwssim)
    residual_cwssim = residual_cwssim[pad_size_x: residual_cwssim.shape[0]-pad_size_x, pad_size_y:residual_cwssim.shape[1]-pad_size_y]
    #visualize_results(residual_cwssim*3, y_valid, "cwssim")

    #residual = (residual_cwssim + residual_ssim) / 2
    residual = residual_cwssim

    return residual


def ssim_metric (x_valid, y_valid, pad_size_x, pad_size_y):
    residual = (1 - ssim(x_valid, y_valid, win_size=11, full=True, data_range=1.)[1])
    residual = residual[pad_size_x: residual.shape[0]-pad_size_x, pad_size_y:residual.shape[1]-pad_size_y]
    return residual


def l2_metric (x_valid, y_valid, pad_size_x, pad_size_y):
    #residual = np.square(x_valid - y_valid)
    residual = np.abs(x_valid - y_valid)
    residual = residual[pad_size_x: residual.shape[0]-pad_size_x, pad_size_y:residual.shape[1]-pad_size_y]
    return residual