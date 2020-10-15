import cv2
import os
import numpy as np
base_folder = '/home/projects/data/validation/'
alpha=0.7

list_imgs = os.listdir(base_folder+'img')
# list_imgs = [os.path.join(base_folder+'img',img) for img in list_imgs]

list_segs = os.listdir(base_folder+'seg')
list_segs = [os.path.join(base_folder+'seg',seg) for seg in list_segs]


for img in list_imgs:
    fullpath2img = os.path.join(base_folder+'img',img)
    fullpath2seg = os.path.join(base_folder+'seg',img)
    img_arr = cv2.imread(fullpath2img)
    seg_arr = cv2.imread(fullpath2seg)
    for i in range(1,45):
        print("no", i,  np.max(seg_arr), len(np.where(seg_arr==i)[0]))



    show_image = np.floor((1-alpha)*img_arr) + np.floor(alpha*seg_arr*255)
    show_image.astype(np.uint8)
    base_save_folder = '/home/projects/src/refineData/outputs/labelTest/'
    if not os.path.isdir(base_save_folder):
        os.makedirs(base_save_folder)
    file2save = base_save_folder + '/{}'.format(img)
    cv2.imwrite(file2save, show_image)