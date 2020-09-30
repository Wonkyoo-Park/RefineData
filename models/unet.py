# import numpy as np
# import os
# import skimage.io as io
# import skimage.transform as trans
# import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import numpy as np
# from skimage.morphology import


from tensorflow.keras import backend as k
from tensorflow import keras


# def iou_metric(y_true_in, y_pred_in, print_table=False):
#     labels = label(y_true_in > 0.5)
#     y_pred = label(y_pred_in > 0.5)
#
#     true_objects = len(np.unique(labels))
#     pred_objects = len(np.unique(y_pred))
#
#     intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
#
#     # Compute areas (needed for finding the union between all objects)
#     area_true = np.histogram(labels, bins=true_objects)[0]
#     area_pred = np.histogram(y_pred, bins=pred_objects)[0]
#     area_true = np.expand_dims(area_true, -1)
#     area_pred = np.expand_dims(area_pred, 0)
#
#     # Compute union
#     union = area_true + area_pred - intersection
#
#     # Exclude background from the analysis
#     intersection = intersection[1:, 1:]
#     union = union[1:, 1:]
#     union[union == 0] = 1e-9
#
#     # Compute the intersection over union
#     iou = intersection / union
#
#     # Precision helper function
#     def precision_at(threshold, iou):
#         matches = iou > threshold
#         true_positives = np.sum(matches, axis=1) == 1  # Correct objects
#         false_positives = np.sum(matches, axis=0) == 0  # Missed objects
#         false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
#         tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
#         return tp, fp, fn
#
#     # Loop over IoU thresholds
#     prec = []
#     if print_table:
#         print("Thresh\tTP\tFP\tFN\tPrec.")
#     for t in np.arange(0.5, 1.0, 0.05):
#         tp, fp, fn = precision_at(t, iou)
#         if (tp + fp + fn) > 0:
#             p = tp / (tp + fp + fn)
#         else:
#             p = 0
#         if print_table:
#             print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
#         prec.append(p)
#
#     if print_table:
#         print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
#     return np.mean(prec)
#
#
# def iou_metric_batch(y_true_in, y_pred_in):
#     batch_size = y_true_in.shape[0]
#     metric = []
#     for batch in range(batch_size):
#         value = iou_metric(y_true_in[batch], y_pred_in[batch])
#         metric.append(value)
#     return np.array(np.mean(metric), dtype=np.float32)
#
#
# def my_iou_metric(label, pred):
#     metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
#     return metric_value


def build_model(batch, pretrained_weights=False, input_size=(512, 512, 3,)):
    inputs = Input(input_size)
    conv1_1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_1)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_1)
    # drop5 = Dropout(0.5)(conv5)
    up6 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv5)
    merge6 = concatenate([up6, conv4], axis=3)
    conv6_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6_1)
    up7 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv6)
    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_1)
    up8 = Conv2DTranspose(16, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_1)
    up9 = Conv2DTranspose(16, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9_1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_1)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', my_iou_metric])

    # model.summary()

    if (pretrained_weights):
        model.load_weights('log1/only_weights.h5')

    return model