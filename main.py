import os
from labelme.util import *
import argparse

BASE_FOLDER = "C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/"
IMAGE_FOLDER = os.path.join(BASE_FOLDER,'img')
SEG_FOLDER = os.path.join(BASE_FOLDER,'labelme')
BOX_FOLDER = os.path.join(BASE_FOLDER,'labelimg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a new image dataset by cropping and producing segmentation images.')
    parser.add_argument('--input_image_dir', '-i', type=str, help='an input folder to original dataset')
    parser.add_argument('--input_xml_dir', '-x', type=str, help='an input folder including xml files of bboxes')
    parser.add_argument('--input_json_dir', '-j', type=str, help='an input folder including json files of segmentation')
    parser.add_argument('--input_xml_bbox_file', '-xf', default="C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/labelimg/_2737476_orig[1].xml"
                        ,type=str, help='an input file path of xml file to bboxes for test')
    parser.add_argument('--input_json_seg_file', '-jf', default="C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/labelme/_2737476_orig[1].json",
                        type=str, help='an input file path of json file of segmentation for test')
    parser.add_argument('--output_dir', '-o', type=str, help='an input folder for refined data to be saved')
    args = parser.parse_args()
    if args.input_image_dir is None:
        dict_name2=dict()
        name2pixelValue = pd.read_csv("labelme/name_list_with_pixel_value.csv", index_col=0, skiprows=0).T.to_dict()
        original_image, label_image = draw_label_png(args.input_json_seg_file,name2pixelValue)
        cropped_ori_img, cropped_lbl_img = crop_label_png(args.input_xml_bbox_file, original_image, label_image)
        show_img(cropped_ori_img)
        show_img(cropped_lbl_img)
    else:
        list_xml_files = os.listdir(args.input_xml_dir)
        list_json_files = os.listdir(args.input_json_dir)

        i=0
        for afile in list_xml_files:
            basename = afile.split('.')[0]
            jsonfilename = basename + '.json'
            if jsonfilename in list_json_files:
                full_path2xml = os.path.join(args.input_xml_dir,afile)
                full_path2json = os.path.join(args.input_json_dir,jsonfilename)
                name2pixelValue = pd.read_csv("labelme/name_list_with_pixel_value.csv", index_col=0,skiprows=0).T.to_dict()
                original_image, label_image = draw_label_png(full_path2json, name2pixelValue)
                cropped_ori_img, cropped_lbl_img = crop_label_png(full_path2xml, original_image, label_image)
                path2save_orig_img = args.output_dir + "/img/"
                path2save_seg_img = args.output_dir + "/seg/"
                if not os.path.isdir(path2save_orig_img):
                    os.makedirs(path2save_orig_img,0o777)
                if not os.path.isdir(path2save_seg_img):
                    os.makedirs(path2save_seg_img,0o777)
                dst_path2save_orig_img = path2save_orig_img + '{:08d}.png'.format(i)
                dst_path2save_seg_img = path2save_seg_img + '{:08d}.png'.format(i)
                save_img(dst_path2save_orig_img,cropped_ori_img)
                save_img(dst_path2save_seg_img,cropped_lbl_img)
                i+=1

                # show_img(cropped_ori_img)
                # show_img(cropped_lbl_img)





