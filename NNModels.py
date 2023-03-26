from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, ZeroPadding2D, Cropping2D, LeakyReLU, Layer, InputSpec
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Input, Add, Concatenate, Lambda
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.models import Model


def Model_noise_skip(input_shape=(None, None, None), latent_dim=300):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv1")(input)
    X_skip_2 = Conv2D(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)

    X_lat = Conv2D(512, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                   kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(3, (4, 4), activation='linear', padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)

    return Model(inputs=input, outputs=X)

def Model_noise_skip_color_only(input_shape=(None, None, 2)):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv1")(input)
    X_skip_2 = Conv2D(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)

    X_lat = Conv2D(512, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                   kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(2, (4, 4), activation='linear', padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)

    return Model(inputs=input, outputs=X)

def Model_noise_skip_big(input_shape=(None, None, None)):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv1")(input)
    X_skip_2 = Conv2D(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                      kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)

    X_lat = Conv2D(1024, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                   kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(3, (4, 4), activation='linear', padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)

    return Model(inputs=input, outputs=X)

def Model_noise_skip_med(input_shape=(None, None, None)):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv1")(input)
    X_skip_2 = Conv2D(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                      kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)

    X_lat = Conv2D(768, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                   kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(3, (4, 4), activation='linear', padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)

    return Model(inputs=input, outputs=X)

def Model_noise_skip_3x(input_shape=(None, None, None)):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv1")(input)
    X_skip_2 = Conv2D(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                      kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)

    X_lat = Conv2D(1536, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                   kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(3, (4, 4), activation='linear', padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)

    return Model(inputs=input, outputs=X)

def Model_noise_skip_bigger(input_shape=(None, None, None)):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv1")(input)
    X_skip_2 = Conv2D(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)

    X_lat = Conv2D(2048, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                   kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(3, (4, 4), activation='linear', padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)

    return Model(inputs=input, outputs=X)

def Model_noise_skip_bigger_old(input_shape=(None, None, None)):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv1")(input)
    X_skip_2 = Conv2D(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                      kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                      kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)

    X_lat = Conv2D(2048, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                   kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4), activation=LeakyReLU(), padding='same', strides=4,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4), activation=LeakyReLU(), padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(3, (4, 4), activation='linear', padding='same', strides=2,
                        kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)

    return Model(inputs=input, outputs=X)

