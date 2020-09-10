## Tensorflow record (.tfrecord) File I/O

This is a document of tfrecord with a customized example to our OCR model

※ The TFRecord  format is briefly documented here, and described as the recommended format for feeding data into Tensorflow here and here.
This library facilitates producing data in the TFRecord format directly in node.js. The library is not "official" - it is not part of Tensorflow,
 and it is not implemented by the Tensorflow team.
 
### code

#### install dependencies

```
matplotlib
tensorflow2.x
numpy
```


### API
the example below covers recommended API usage
#### write images to tfrecord for segmentation model

```
in 'generate_tfrecords.py'.
write_tfrecords(i,path2base,path2imgs,path2segs,list_images_inatfrecord)
```

```
@params
i : a list of normal class on the top of scene,
path2base : a base folder directory having subdirectories of original images folder and segmentation images folder,
path2imgs : a base folder directory of original images,
path2segs : a base folder directory of segmentation images,
list_images_inatfrecord : a list of image file names
```

```
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

```

#### read and load TFRecord for 'cardchecker' classification model
```
read_tfrecord(imageTFRecords,num_parallel_reads=8,shuffle_buffer_size=8,batch_size=16)
```

```
@params
imageTFRecords : a list of target tfrecord files
num_parallel_reads : Number of threads to work 
shuffle_buffer_size : size of shuffle buffer
batch_size : Number of batch images
```

```
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
```

#### Augment input data after loading data
```
data = apply_aug_train(image_features, args)
```

```
@params
image_features : decoded images data from tfrecord
args : set of arguments containing information, batch_size, target_width, target_height

@output
augmented data of batch size
```

```
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
        if rand2 > 0.5:
            offset = int(np.floor(50 * random.random()))
            if decode_img.shape[1] >= height and decode_img.shape[2] >= width:
                decode_img = tf.image.crop_to_bounding_box(decode_img, 0, offset, height, width - offset)
                decode_img = tf.image.pad_to_bounding_box(decode_img, 0, int(np.floor(offset / 2)), height, width)
                decode_seg = tf.image.crop_to_bounding_box(decode_seg, 0, offset, height, width - offset)
                decode_seg = tf.image.pad_to_bounding_box(decode_seg, 0, int(np.floor(offset / 2)), height, width)
            else:
                decode_img = tf.image.resize_with_crop_or_pad(decode_img, height, width)
                decode_seg = tf.image.resize_with_crop_or_pad(decode_seg, height, width)

        else:
            decode_img = tf.image.resize_with_crop_or_pad(decode_img, height, width)
            decode_seg = tf.image.resize_with_crop_or_pad(decode_seg, height, width)

        batch_img.append(decode_img)
        batch_seg.append(decode_seg)

    batch_img=tf.convert_to_tensor(np.asarray(batch_img),dtype=tf.float32)
    batch_seg=tf.convert_to_tensor(np.asarray(batch_seg),dtype=tf.float32)

    batch_seg=np.asarray(batch_seg)

    data = dict()
    data['img']=batch_img
    data['seg']=batch_seg
    # data['label_falling'] = decode_label_falling
    return data
```