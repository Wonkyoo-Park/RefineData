# -*- coding: utf-8 -*-
import os
from labelme.util import *
import argparse
import pandas as pd
import pathlib

BASE_FOLDER = "C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/"
IMAGE_FOLDER = os.path.join(BASE_FOLDER,'img')
SEG_FOLDER = os.path.join(BASE_FOLDER,'labelme')
BOX_FOLDER = os.path.join(BASE_FOLDER,'labelimg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a new image dataset by cropping and producing segmentation images.')
    # parser.add_argument('--input_image_dir', '-i', default="C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/img/",
    parser.add_argument('--mode', '-m', default="series",
                        type=str, help='one of [series, one]')
    parser.add_argument('--input_base_dir', '-b', default="/home/projects/data/train/",
                        type=str, help='an base folder to original dataset')
    parser.add_argument('--input_image_dir', '-i', default="/home/projects/data/data_for_demo/blepharitis/images/",
                        type=str, help='an input folder to original dataset')
    parser.add_argument('--input_xml_dir', '-x', default="/home/projects/data/data_for_demo/blepharitis/labelimg/",
                        type=str, help='an input folder including xml files of bboxes')
    parser.add_argument('--input_json_dir', '-j', default="/home/projects/data/data_for_demo/blepharitis/labelme/",
                        type=str, help='an input folder including json files of segmentation')
    parser.add_argument('--input_xml_bbox_file', '-xf', default="C:/Users/dylee/Documents/카카오톡 받은 파일/해당에러파일들/해당에러파일들/1t[1].xml"
                        ,type=str, help='an input file path of xml file to bboxes for test')
    parser.add_argument('--input_json_seg_file', '-jf', default="C:/Users/dylee/Documents/카카오톡 받은 파일/해당에러파일들/해당에러파일들/1t[1].json",
                        type=str, help='an input file path of json file of segmentation for test')
    parser.add_argument('--output_dir', '-o', default="/home/projects/data/train/",type=str, help='an input folder for refined data to be saved')
    args = parser.parse_args()



    if args.input_image_dir is None:
        dict_name2=dict()
        name2pixelValue = pd.read_csv("labelme/name_list_with_pixel_value.csv", index_col=0, skiprows=0).T.to_dict()
        original_image, label_image = draw_label_png(args.input_json_seg_file,name2pixelValue)
        cropped_ori_img, cropped_lbl_img = crop_label_png(args.input_xml_bbox_file, original_image, label_image)
        show_img(cropped_ori_img)
        show_img(cropped_lbl_img)
    else:
        if args.mode == 'one':
            list_xml_files = os.listdir(args.input_xml_dir)
            list_json_files = os.listdir(args.input_json_dir)
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            i=0
            for afile in list_xml_files:
                basename = afile.split('.')[0]
                jsonfilename = basename + '.json'
                if jsonfilename in list_json_files:
                    full_path2xml = os.path.join(args.input_xml_dir,afile)
                    full_path2json = os.path.join(args.input_json_dir,jsonfilename)

                    name2pixelValue = pd.read_csv(BASE_DIR+"/../labelme/name_list_with_pixel_value.csv", index_col=0,skiprows=0).T.to_dict()
                    original_image, label_image = draw_label_png(full_path2json, name2pixelValue)
                    list_cropped_ori_img, list_cropped_lbl_img = crop_label_png(full_path2xml, original_image, label_image)
                    for cropped_ori_img, cropped_lbl_img in zip(list_cropped_ori_img, list_cropped_lbl_img):
                        path2save_orig_img = args.output_dir + "/img/"
                        path2save_seg_img = args.output_dir + "/seg/"
                        if not os.path.isdir(path2save_orig_img):
                            os.makedirs(path2save_orig_img,0o777)
                        if not os.path.isdir(path2save_seg_img):
                            os.makedirs(path2save_seg_img,0o777)
                        dst_path2save_orig_img = path2save_orig_img + '{:08d}.png'.format(i)
                        dst_path2save_seg_img = path2save_seg_img + '{:08d}.png'.format(i)
                        while(os.path.isfile(dst_path2save_orig_img)):
                            i += 1
                            dst_path2save_orig_img = path2save_orig_img + '{:08d}.png'.format(i)
                            dst_path2save_seg_img = path2save_seg_img + '{:08d}.png'.format(i)

                        save_img(dst_path2save_orig_img,cropped_ori_img)
                        save_img(dst_path2save_seg_img,cropped_lbl_img)
                        i+=1

                    # show_img(cropped_ori_img)
                    # show_img(cropped_lbl_img)
        elif args.mode=='series':
            input_folders = os.listdir(args.input_base_dir)
            input_folders = [os.path.join(args.input_base_dir,afoder) for afoder in input_folders if not afoder in ['img','seg','tfrecords']]

            list_xml_files = []
            list_json_files = []
            for base_unit in input_folders:
                sub_dirs = os.listdir(base_unit)
                for _sub in sub_dirs:
                    try:
                        if _sub.split('_')[1] == 'labelme':
                            tmp_json_list = os.listdir(os.path.join(base_unit,_sub))
                            tmp_json_list = [os.path.join(base_unit,_sub,afile) for afile in tmp_json_list]
                            list_json_files += tmp_json_list
                        elif _sub.split('_')[1] == 'labelimg':
                            tmp_xml_list = os.listdir(os.path.join(base_unit, _sub))
                            tmp_xml_list = [os.path.join(base_unit,_sub,afile) for afile in tmp_xml_list]
                            list_xml_files += tmp_xml_list
                    except:
                        pass

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            i = 0
            for afile in list_xml_files:
                fullpath2parentfolder = pathlib.Path(afile).parent.parent.absolute()
                parentfolder = pathlib.Path(afile).parent.parent._parts[-1]
                file_name=pathlib.Path(afile).stem
                full_path2json = os.path.join(fullpath2parentfolder,parentfolder+'_labelme/'+file_name+'.json')

                # basename = afile.split('.')[0]
                # jsonfilename = basename + '.json'
                # fullpath2folder=pathlib.Path(afile).parent.absolute()




                # if jsonfilename in list_json_files:
                if os.path.isfile(full_path2json):
                    full_path2xml = afile
                    # full_path2xml = os.path.join(args.input_xml_dir, afile)
                    # full_path2json = os.path.join(args.input_json_dir, jsonfilename)

                    name2pixelValue = pd.read_csv(BASE_DIR + "/../labelme/name_list_with_pixel_value.csv", index_col=0,
                                                  skiprows=0).T.to_dict()
                    original_image, label_image = draw_label_png(full_path2json, name2pixelValue)
                    list_cropped_ori_img, list_cropped_lbl_img = crop_label_png(full_path2xml, original_image,
                                                                                label_image)
                    for cropped_ori_img, cropped_lbl_img in zip(list_cropped_ori_img, list_cropped_lbl_img):
                        path2save_orig_img = args.output_dir + "/img/"
                        path2save_seg_img = args.output_dir + "/seg/"
                        if not os.path.isdir(path2save_orig_img):
                            os.makedirs(path2save_orig_img, 0o777)
                        if not os.path.isdir(path2save_seg_img):
                            os.makedirs(path2save_seg_img, 0o777)
                        dst_path2save_orig_img = path2save_orig_img + '{:08d}.png'.format(i)
                        dst_path2save_seg_img = path2save_seg_img + '{:08d}.png'.format(i)
                        while (os.path.isfile(dst_path2save_orig_img)):
                            i += 1
                            dst_path2save_orig_img = path2save_orig_img + '{:08d}.png'.format(i)
                            dst_path2save_seg_img = path2save_seg_img + '{:08d}.png'.format(i)

                        save_img(dst_path2save_orig_img, cropped_ori_img)
                        save_img(dst_path2save_seg_img, cropped_lbl_img)
                        i += 1

                    # show_img(cropped_ori_img)
                    # show_img(cropped_lbl_img)




