import os

import numpy as np
import cv2
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms as T 

from utils.post_processing import segment, cleanContours, findBoundingBox
from utils.data_org import load_img


CAMERA_ORI = "horizontal"
SHOE_FACTOR = "shoe"

# calculate the number of images currently in the data folder
path = f"./data/{CAMERA_ORI}/{SHOE_FACTOR}"
N = int((len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])) / 3)

# import models
fcn = models.segmentation.fcn_resnet101(weights=models.segmentation.FCN_ResNet101_Weights.DEFAULT).eval()
dlab = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()

# final image list for report generation purposes
im_list = []

for i in np.arange(10):
    path = f'data/{CAMERA_ORI}/{SHOE_FACTOR}/color_{i}.jpeg'
    print(f'processing image: {path}')
    
    img_rgb, img_np = load_img(path, CAMERA_ORI)

    mask_b, mask_rgb = segment(img_rgb, img_np, dlab, visual=False)
    mask_b, img_cnt = cleanContours(img_np, mask_b, cnt_approx=False, visual=False)
    bbs,img_bb = findBoundingBox(img_cnt, mask_b)

    cv2.imshow("original image", img_np)
    cv2.imshow("image with bounding box and contour", img_bb)
    cv2.waitKey(0)

    im_list.append(img_rgb)

# im_list[0].save(f'./results.pdf', save_all=True, append_images=im_list[1:])


