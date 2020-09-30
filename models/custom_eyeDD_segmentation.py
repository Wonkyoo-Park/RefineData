import sys
sys.path.append('../')
#from models.ops import *

import numpy as np
import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D,MaxPooling2D, MaxPool2D, Activation, concatenate, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects, get_file
from tensorflow import keras
# from models.util_rec import _transform

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)
    left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)
    right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)
    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x



def build_model(include_top=True,batch=2,height=400, width=400, color=True, filters=64, pooling='avg',classes1=3,classes2=2):
    inputs = keras.layers.Input((height, width, 3 if color else 1), batch_size=batch)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=10, squeeze=48, expand=192)
    x = fire_module(x, fire_id=11, squeeze=48, expand=192)
    x = fire_module(x, fire_id=12, squeeze=64, expand=256)
    x = fire_module(x, fire_id=13, squeeze=64, expand=256)

    model = Model(inputs, x, name='squeezenet')


    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH,
    #                                 cache_subdir='models')
    #     else:
    #         weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                 WEIGHTS_PATH_NO_TOP,
    #                                 cache_subdir='models')
    #
    #     model.load_weights(weights_path)


    # if include_top:
        # It's not obvious where to cut the network...
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    # model = tf.keras.Sequential([
    #     x,
    #     layers.Dense(image_data.num_classes, activation='softmax')
    # ])


    x = Dropout(0.5, name='drop9')(x)
    y = Dropout(0.5, name='drop10')(x)
    x = Conv2D(classes1, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='location_output1')(x)


    y = Conv2D(classes2, (1, 1), padding='valid', name='conv11')(y)
    y = Activation('relu', name='relu_conv11')(y)
    y = GlobalAveragePooling2D()(y)
    y = Activation('softmax', name='falling_output2')(y)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #         y = GlobalAveragePooling2D()(y)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)
    #         y = GlobalAveragePooling2D()(y)
    #     elif pooling == None:
    #         pass
    #     else:
    #         raise ValueError("Unknown argument for 'pooling'=" + pooling)




    cutom_model = Model(inputs, [x,y], name='squeezenet')

    # weights_path = "output\\checkpoints\\eye\\generator_scale_300.h5'
    # cutom_model.load_weights(weights_path)

    return cutom_model