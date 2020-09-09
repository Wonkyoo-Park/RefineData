import imgviz
import matplotlib.pyplot as plt
import PIL.Image
from labelme.LabelFile import *
import math
import uuid
import cv2
import pandas as pd
import xml.etree.ElementTree as elemTree


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id['pixelValue']
        ins[mask] = ins_id

    return cls, ins

# def lblsave(filename, lbl):
def lblsave(lbl):
    # if osp.splitext(filename)[1] != ".png":
    #     filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        # lbl_pil.save(filename)
        lbl_array = np.array(lbl_pil)
        # cv2.imshow("label", lbl_array*10)
        # cv2.waitKey(0)
        return lbl_array
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format."
        )

def draw_label_png(label_me_json,name2pixelValue):
    label_file = LabelFile(label_me_json)
    img = img_data_to_arr(label_file.imageData)
    name2pixelValue["_background_"] = 0
    label_name_to_value = name2pixelValue
    # label_name_to_value = {"_background_": 0}
    for shape in sorted(label_file.shapes, key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = shapes_to_label(
        img.shape, label_file.shapes, label_name_to_value
    )
    lbl = lblsave(lbl)
    return img, lbl


def crop_label_png(label_xml,original_image,label_image):
    tree = elemTree.parse(label_xml)
    bbox = tree.getroot().find('object').find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    cropped_ori_img = original_image[ymin:ymax,xmin:xmax]
    cropped_lbl_img = label_image[ymin:ymax,xmin:xmax]
    return cropped_ori_img, cropped_lbl_img

def show_img(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)

def save_img(path2img,img):
    cv2.imwrite(path2img,img)



if __name__ == "__main__":
    label_json = "C:/Users/dylee/Documents/alphado/dataset/refined_dataset/keratitis/labelme/_2737476_orig[1].json"
    dict_name2=dict()
    name2pixelValue = pd.read_csv("name_list_with_pixel_value.csv", index_col=0, skiprows=0).T.to_dict()
    draw_label_png(label_json,name2pixelValue)




