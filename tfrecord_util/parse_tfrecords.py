import tensorflow as tf
import random
import numpy as np
import argparse
import os

def apply_aug_train(image_features,args):
    batch_size=args.batch_size
    width=args.target_width
    height=args.target_height
    # try:
    decode_height = tf.reshape(image_features['image/height'], (batch_size, 1))
    decode_width = tf.reshape(image_features['image/width'], (batch_size, 1))
    batch_img=[]
    batch_seg = []
    a=int(decode_height[1])
    for i in range(batch_size):
        decode_img = tf.io.decode_raw(image_features['image/image_raw'][i], tf.uint8)
        decode_seg = tf.io.decode_raw(image_features['image/segmentation_raw'][i], tf.uint8)
        decode_img = tf.reshape(decode_img, (1, int(decode_height[i]), int(decode_width[i]), 3))
        decode_seg = tf.reshape(decode_seg, (1, int(decode_height[i]), int(decode_width[i]), 3))

        rand0 = random.random()
        if rand0 > 0.5:
            decode_img = tf.image.flip_left_right(decode_img)
            decode_seg = tf.image.flip_left_right(decode_seg)
        rand1 = random.random()
        if rand1 > 0.5:
            decode_img = tf.image.flip_up_down(decode_img)
            decode_seg = tf.image.flip_up_down(decode_seg)


        if rand1 <= 0.33:
            decode_img = tf.image.rot90(decode_img)
            decode_seg = tf.image.rot90(decode_seg)
        elif rand1 <= 0.67:
            decode_img = tf.image.rot90(decode_img)
            decode_img = tf.image.rot90(decode_img)
            decode_img = tf.image.rot90(decode_img)
            decode_seg = tf.image.rot90(decode_seg)
            decode_seg = tf.image.rot90(decode_seg)
            decode_seg = tf.image.rot90(decode_seg)

        rand2 = random.random()
        # if rand2 > 0.5:
        #     offset = int(np.floor(50 * random.random()))
        #     if decode_img.shape[1] >= height and decode_img.shape[2] >= width:
        #         decode_img = tf.image.crop_to_bounding_box(decode_img, 0, offset, height, width - offset)
        #         decode_img = tf.image.pad_to_bounding_box(decode_img, 0, int(np.floor(offset / 2)), height, width)
        #         decode_seg = tf.image.crop_to_bounding_box(decode_seg, 0, offset, height, width - offset)
        #         decode_seg = tf.image.pad_to_bounding_box(decode_seg, 0, int(np.floor(offset / 2)), height, width)
        #     else:
        #         decode_img = tf.image.resize_with_crop_or_pad(decode_img, height, width)
        #         decode_seg = tf.image.resize_with_crop_or_pad(decode_seg, height, width)
        #
        # else:
        #     decode_img = tf.image.resize_with_crop_or_pad(decode_img, height, width)
        #     decode_seg = tf.image.resize_with_crop_or_pad(decode_seg, height, width)
        decode_img = tf.image.resize(decode_img, [height, width])
        decode_seg = tf.image.resize(decode_seg, [height, width],method='nearest')

        batch_img.append(decode_img)
        batch_seg.append(decode_seg)

    batch_img=tf.concat(batch_img,axis=0)
    batch_seg=tf.concat(batch_seg,axis=0)


    # batch_img=tf.convert_to_tensor(np.asarray(batch_img),dtype=tf.float32)
    # batch_seg=tf.convert_to_tensor(np.asarray(batch_seg),dtype=tf.float32)

    batch_seg=np.asarray(batch_seg)

    data = dict()
    data['img']=batch_img
    data['seg']=batch_seg
    # data['label_falling'] = decode_label_falling
    return data

def apply_validation(image_features,args):
    batch_size = args.val_batch_size
    width = args.target_width
    height = args.target_height

    decode_height = tf.reshape(image_features['image/height'], (batch_size, 1))
    decode_width = tf.reshape(image_features['image/width'], (batch_size, 1))
    batch_img = []
    batch_seg = []

    for i in range(batch_size):
        decode_img = tf.io.decode_raw(image_features['image/image_raw'][i], tf.uint8)
        decode_seg = tf.io.decode_raw(image_features['image/segmentation_raw'][i], tf.uint8)
        decode_img = tf.reshape(decode_img, (1, int(decode_height[i]), int(decode_width[i]), 3))
        decode_seg = tf.reshape(decode_seg, (1, int(decode_height[i]), int(decode_width[i]), 3))
        decode_img = tf.image.resize(decode_img,[height,width])
        decode_seg = tf.image.resize(decode_seg,[height,width],method='nearest')
        batch_img.append(decode_img)
        batch_seg.append(decode_seg)


    batch_img = tf.concat(batch_img, axis=0)
    batch_seg = tf.concat(batch_seg, axis=0)

    data = dict()
    data['img']=batch_img
    data['seg']=batch_seg
    return data

def apply_only_img_validation(image_features,args):
    batch_size = args.val_batch_size
    width = args.target_width
    height = args.target_height

    decode_height = tf.reshape(image_features['image/height'], (batch_size, 1))
    decode_width = tf.reshape(image_features['image/width'], (batch_size, 1))
    batch_img = []
    batch_seg = []

    for i in range(batch_size):
        decode_img = tf.io.decode_raw(image_features['image/image_raw'][i], tf.uint8)
        # decode_seg = tf.io.decode_raw(image_features['image/segmentation_raw'][i], tf.uint8)
        decode_img = tf.reshape(decode_img, (1, int(decode_height[i]), int(decode_width[i]), 3))
        # decode_seg = tf.reshape(decode_seg, (1, int(decode_height[i]), int(decode_width[i]), 3))
        decode_img = tf.image.resize(decode_img,[height,width])
        # decode_seg = tf.image.resize(decode_seg,[height,width])
        batch_img.append(decode_img)
        # batch_seg.append(decode_seg)


    batch_img = tf.concat(batch_img, axis=0)
    # batch_seg = tf.concat(batch_seg, axis=0)

    data = dict()
    data['img']=batch_img
    # data['seg']=batch_seg
    return data


def read_tfrecord(imageTFRecords,num_parallel_reads=8,shuffle_buffer_size=8,batch_size=16):
    tfrecordFiles = tf.data.Dataset.list_files(imageTFRecords)
    dataset = tfrecordFiles.interleave(tf.data.TFRecordDataset,cycle_length=num_parallel_reads,num_parallel_calls=2)


    image_feature_description = {
        'image/image_raw':tf.io.FixedLenFeature((),tf.string),
        'image/segmentation_raw':tf.io.FixedLenFeature((),tf.string),
        'image/format': tf.io.FixedLenFeature((), tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }

    def _partse_image_function(example_proto):
        data_features =tf.io.parse_single_example(example_proto,image_feature_description)
        data=data_features

        return data


    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(_partse_image_function,num_parallel_calls=2).batch(batch_size=batch_size,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=96)
    # dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset

def read_record_validation(imageTFRecord,num_parallel_reads=8,shuffle_buffer_size=8,batch_size=16,epoch=10):
    tfrecordFiles = tf.data.Dataset.list_files(imageTFRecord)
    dataset = tfrecordFiles.interleave(tf.data.TFRecordDataset,cycle_length=num_parallel_reads,num_parallel_calls=tf.data.experimental.AUTOTUNE)


    image_feature_description = {
        'image/image_raw':tf.io.FixedLenFeature((),tf.string),
        'image/segmentation_raw':tf.io.FixedLenFeature((),tf.string),
        'image/format': tf.io.FixedLenFeature((), tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }

    def _partse_image_function(example_proto):
        data_features =tf.io.parse_single_example(example_proto,image_feature_description)

        data=data_features
        return data


    # dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(_partse_image_function,num_parallel_calls=2).batch(batch_size=batch_size,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=96)
    # dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset

def read_record_only_img_validation(imageTFRecord,num_parallel_reads=8,shuffle_buffer_size=8,batch_size=16,epoch=10):
    tfrecordFiles = tf.data.Dataset.list_files(imageTFRecord)
    dataset = tfrecordFiles.interleave(tf.data.TFRecordDataset,cycle_length=num_parallel_reads,num_parallel_calls=tf.data.experimental.AUTOTUNE)


    image_feature_description = {
        'image/image_raw':tf.io.FixedLenFeature((),tf.string),
        'image/format': tf.io.FixedLenFeature((), tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }

    def _partse_image_function(example_proto):
        data_features =tf.io.parse_single_example(example_proto,image_feature_description)

        data=data_features
        return data


    # dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(_partse_image_function,num_parallel_calls=2).batch(batch_size=batch_size,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=96)
    # dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2tfrecords", default='C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/dst/tfrecords/',
                        type=str,help="path to the folders containing tfrecord files")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--target_width", default=512, type=int, help="target width of input image")
    parser.add_argument("--target_height", default=512, type=int, help="target width of input image")
    args = parser.parse_args()
    if not os.path.isdir(args.path2tfrecords):
        raise ValueError(
            "Folder path might be wrong"
        )
    list_tfrecords = os.listdir(args.path2tfrecords)
    list_tfrecords = [os.path.join(args.path2tfrecords,path2tfrecord) for path2tfrecord in list_tfrecords]
    dataset = read_tfrecord(list_tfrecords[:-1], batch_size=args.batch_size)
    validation_dataset = read_record_validation(list_tfrecords[-1], batch_size=args.batch_size)
    for i, image_features in enumerate(dataset):
        data = apply_aug_train(image_features, args)
    for i, image_features in enumerate(validation_dataset):
        data = apply_validation(image_features, args)


