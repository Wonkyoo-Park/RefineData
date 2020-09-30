import os
import argparse
import tensorflow
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecords(i,path2base,path2imgs,path2segs,list_images_inatfrecord):
    if not os.path.isdir(path2base +'/tfrecords'):
        os.makedirs(path2base +'/tfrecords')

    with tf.io.TFRecordWriter(path2base +'/tfrecords/{:04d}_images.tfrecords'.format(i)) as writer:
        def image_example(img_raw, seg_raw, height, width):
            feature = {
                'image/image_raw': _bytes_feature(img_raw),
                'image/segmentation_raw': _bytes_feature(seg_raw),
                'image/format': _bytes_feature(b'png'),
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        for name2image in list_images_inatfrecord:
            path2imgfile = os.path.join(path2imgs,name2image)
            path2segfile = os.path.join(path2segs,name2image)
            tfimg = tf.keras.preprocessing.image.load_img(path2imgfile)
            tfseg = tf.keras.preprocessing.image.load_img(path2segfile)

            img_array = tf.keras.preprocessing.image.img_to_array(tfimg).astype(np.uint8)
            seg_array = tf.keras.preprocessing.image.img_to_array(tfseg).astype(np.uint8)

            height = img_array.shape[0]
            width = img_array.shape[1]

            img_bytes = img_array.tostring()
            seg_bytes = seg_array.tostring()

            tf_example = image_example(img_bytes, seg_bytes, height=height, width=width)
            writer.write(tf_example)

if __name__ == "__main__":
    base_folder = "C:\\Users\\dylee\\PycharmProjects\\escalatorEventDetection\\falling_dnn\\ocr_onnx\\datagenerator\\dataset\\tmp\\"
    list_validation = os.listdir(base_folder)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default='read',type=str,help='one of [write, read]')
    parser.add_argument("--target_task", default='card_checker', type=str, help='one of [card_checler, NonID_checker, condition_checker]')
    parser.add_argument("--path2base", default='C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/dst/', type=str)

    args = parser.parse_args()

    list_images = os.listdir(os.path.join(args.path2base,'img'))
    # list_labels = os.listdir(os.path.join(args.path2base,'seg'))
    Noftfrecords = 4
    NofImagesinfrecord = int(np.floor(len(list_images)/Noftfrecords))

    list_tfrecord_units = []
    for i in range(Noftfrecords):
        if i<Noftfrecords-1:
            list_tfrecord_units.append(list_images[i*NofImagesinfrecord:(i+1)*NofImagesinfrecord])
        else:
            list_tfrecord_units.append(list_images[i*NofImagesinfrecord:])

    if not os.path.isdir(args.path2base):
        os.makedirs(args.path2base)
    for i, list_images_inatfrecord in enumerate(list_tfrecord_units):
            path2imgs = os.path.join(args.path2base,'img')
            path2segs = os.path.join(args.path2base, 'seg')
            write_tfrecords(i,args.path2base,path2imgs,path2segs,list_images_inatfrecord)
