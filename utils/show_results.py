#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np

from PIL import Image


classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[0,0,255],[255,0,0], [255,255,0], [0,128,255],[255,0,255]]
            #,[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            #[64,128,0],[192,128,0],[64,0,128],[192,0,128],
            #[64,128,128],[192,128,128],[0,64,0],[128,64,0],
            #[0,192,0],[128,192,0],[0,64,128]]

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def single_image_visual_result(image, prediction,target_features, alpha=0.5):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    # image = (image * rgb_std + rgb_mean) * 255
    image=np.asarray(image)

    prediction=np.asarray(prediction)


    prediction=prediction[0]


    H, W, C = image.shape
    H, W, nC = prediction.shape
    masks_color = np.zeros(shape=[H, W, C])
    lables_color = np.zeros(shape=[H, W, C])
    numpy_prediction = np.zeros_like(prediction)
    cls = []
    c_i = 1
    result_image=np.zeros_like(image)
    no2name_dict = dict()
    no2name_dict['1']='Third_eyelid_protrude'
    no2name_dict['2']='blepharitis_inflammation'
    no2name_dict['3']='blepharitis_inner_inflammation'
    no2name_dict['4']='gataract'
    no2name_dict['5']='gataract_initial'

    result_dict = dict()
    result_dict['Third_eyelid_protrude']=0
    result_dict['blepharitis_inflammation']=0
    result_dict['blepharitis_inner_inflammation']=0
    result_dict['gataract']=0
    result_dict['gataract_initial']=0

    result_prob_dict = dict()
    result_prob_dict['Third_eyelid_protrude']=0
    result_prob_dict['blepharitis_inflammation']=0
    result_prob_dict['blepharitis_inner_inflammation']=0
    result_prob_dict['gataract']=0
    result_prob_dict['gataract_initial']=0

    for k in range(nC):
        if k in target_features:
            NofValidPixels=0
            maxRatio=0
        # if np.max(label[...,k])> 0 and k> 0:
            # print("no",no, np.max(label[...,k]), len(np.where(label[...,k]>0)[0]))

            for i in range(H):
                for j in range(W):
                    if prediction[i, j,k] > 0.4:
                        numpy_prediction[i, j, k] = 1
                        NofValidPixels+=1
                        if maxRatio<prediction[i, j,k]:
                            maxRatio=prediction[i, j,k]

                    else:
                        numpy_prediction[i, j, k] = 0


                    if numpy_prediction[i, j, k] >0.5 and k>0:
                        masks_color[i, j] = np.array(colormap[c_i])
                        # cls.append(cls_idx)
                    else:
                        masks_color[i, j] = np.array(colormap[0])
            c_i+=1
            masks_color = masks_color.astype(np.uint8)
            show_image = np.zeros(shape=[512, 1024, 3])
            cls = set(cls)
            NofPixels = masks_color.shape[0]*masks_color.shape[1]
            NofValidPixels = NofValidPixels
            print("ratio valid pixels : ", NofValidPixels/NofPixels)
            if NofValidPixels/NofPixels<0.9:
                result_image+= masks_color
                if NofValidPixels/NofPixels>0.001:
                    disease_name=no2name_dict[str(c_i-1)]
                    result_dict[disease_name]=1
                    result_prob_dict[disease_name]=maxRatio
            # else:
            #     result_image += np.zeros_like(masks_color)
            # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
            # show_image = (1-alpha)*image + alpha*lables_color
            show_image[:,:512,:] = image
            show_image[:,512:,:] = masks_color
            # print("no", no, np.max(masks_color), len(np.where(masks_color > 0)[0]))
            # show_image = Image.fromarray(np.uint8(show_image))
            show_image = show_image.astype(np.uint8)
            base_save_folder = '/home/projects/src/refineData/outputs/snapshot/'
            if not os.path.isdir(base_save_folder):
                os.makedirs(base_save_folder)
            cv2.imwrite(base_save_folder+'/one_shot_no_{}.png'.format(k),show_image)
    cv2.imwrite(base_save_folder+'/result_no.png'.format(k),result_image)
            # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
            # cv2.WaitKey(0)
    print(result_dict)
    return result_dict, result_prob_dict, result_image

def visual_result(no, image, prediction,label,alpha=0.5):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    image, label = image.astype(np.uint8), label.astype(np.uint8)
    image = image[0]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = prediction[0]
    numpy_prediction = np.zeros_like(prediction)
    label = label[0]

    H, W, C = image.shape
    H, W, nC = label.shape
    masks_color = np.zeros(shape=[H, W, C])
    lables_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    cls = []
    for k in range(nC):
        # if k in target_features:
        if np.max(label[...,k])> 0 and k> 0:
            # print("no",no, np.max(label[...,k]), len(np.where(label[...,k]>0)[0]))
            for i in range(H):
                for j in range(W):
                    if prediction[i, j,k] > 0.4:
                        numpy_prediction[i, j, k] = 1
                    else:
                        numpy_prediction[i, j, k] = 0
                    cls_idx = label[i, j, k]

                    if cls_idx >0 and k>0:
                        lables_color[i, j] = np.array(colormap[2])
                        # cls.append(cls_idx)
                    else:
                        lables_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] >0.5 and k>0:
                        masks_color[i, j] = np.array(colormap[c_i])
                        # cls.append(cls_idx)
                    else:
                        masks_color[i, j] = np.array(colormap[0])

            masks_color = masks_color.astype(np.int)
            show_image = np.zeros(shape=[512, 1024, 3])
            cls = set(cls)
            # /

            # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
            # show_image = (1-alpha)*image + alpha*lables_color
            show_image[:,:512,:] = image
            show_image[:,512:,:] = masks_color
            print("no", np.max(masks_color), len(np.where(masks_color > 0)[0]))
            # show_image = Image.fromarray(np.uint8(show_image))
            show_image = show_image.astype(np.uint8)
            base_save_folder = '/home/projects/src/refineData/outputs/snapshot/'
            if not os.path.isdir(base_save_folder):
                os.makedirs(base_save_folder)

            cv2.imwrite(base_save_folder + '/snapshot_no_{}_{}.png'.format(no, k), show_image)
    cv2.imwrite(base_save_folder + '/result_no_{}_{}.png'.format(no, k), result_image)
            # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
            # cv2.WaitKey(0)

    # return show_image

def create_image_label_path_generator(images_filepath, labels_filepath):
    image_paths = open(images_filepath).readlines()
    all_label_txts = os.listdir(labels_filepath)
    image_label_paths = []
    for label_txt in all_label_txts:
        label_name = label_txt[:-4]
        label_path = labels_filepath + "/" + label_txt
        for image_path in image_paths:
            image_path = image_path.rstrip()
            image_name = image_path.split("/")[-1][:-4]
            if label_name == image_name:
                image_label_paths.append((image_path, label_path))
    while True:
        random.shuffle(image_label_paths)
        for i in range(len(image_label_paths)):
            yield image_label_paths[i]

def process_image_label(image_path, label_path):
    # image = misc.imread(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # data augmentation here
    # randomly shift gamma
    gamma = random.uniform(0.8, 1.2)
    image = image.copy() ** gamma
    image = np.clip(image, 0, 255)
    # randomly shift brightness
    brightness = random.uniform(0.5, 2.0)
    image = image.copy() * brightness
    image = np.clip(image, 0, 255)
    # image transformation here
    image = (image / 255. - rgb_mean) / rgb_std

    label = open(label_path).readlines()
    label = [np.array(line.rstrip().split(" ")) for line in label]
    label = np.array(label, dtype=np.int)
    label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
    label = label.astype(np.int)

    return image, label


def DataGenerator(train_image_txt, train_labels_dir, batch_size):
    """
    generate image and mask at the same time
    """
    image_label_path_generator = create_image_label_path_generator(
        train_image_txt, train_labels_dir
    )
    while True:
        images = np.zeros(shape=[batch_size, 224, 224, 3])
        labels = np.zeros(shape=[batch_size, 224, 224], dtype=np.float)
        for i in range(batch_size):
            image_path, label_path = next(image_label_path_generator)
            image, label = process_image_label(image_path, label_path)
            images[i], labels[i] = image, label
        yield images, labels
