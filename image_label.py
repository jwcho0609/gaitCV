import os

import numpy as np
import cv2
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms as T 

from utils.post_processing import filter_depth, approx_bounding_box, perform_grabcut, visualize_boundbox


CAMERA_ORI = "horizontal"
SHOE_FACTOR = "shoe"

# calculate the number of images currently in the data folder
path = f"./data/{CAMERA_ORI}/{SHOE_FACTOR}"
N = int((len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])) / 3)

# import models
fcn = models.segmentation.fcn_resnet101(weights=models.segmentation.FCN_ResNet101_Weights.DEFAULT).eval()
dlab = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()


def load_img(path):
    img_rgb = Image.open(path)

    if CAMERA_ORI == 'vertical':
        img_rgb = img_rgb.rotate(90, Image.NEAREST, expand = 1)
    img_np = np.array(img_rgb.convert('RGB'))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_rgb, img_np


# Define the helper function
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        if l == 15:
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(img_rgb, img_np, net, visual=False):
    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([
                    # T.Resize(256), 
                    # T.CenterCrop(224), 
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img_rgb).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    mask_rgb = decode_segmap(om)

    # convert our RGB mask into an image for PDF report generation
    mask_rgb = Image.fromarray(mask_rgb.astype('uint8'),'RGB')
    
    # binary mask conversion
    mask_b = np.sum(mask_rgb, axis=2).astype(np.uint8)
    mask_b[mask_b > 0] = 1

    if visual:
        cv2.imshow('original image', img_np)
        cv2.imshow('mask', mask_b*255)
        cv2.waitKey(0)

    return mask_b, mask_rgb


def cleanContours(img_np, mask, min_area=1000, visual=False):

    # find only the external contours in the mask
    cnts, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of Contours found = " + str(len(cnts)))

    # stacked mask to comply for RGB image dimensions: (N,M,3)
    mask_stack = np.dstack((mask,mask,mask))*255

    # iterate through each contour, and remove those that are less than a specified size
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            mask_stack = cv2.drawContours(mask_stack, [c], -1, (0,0,0), -1)
        else:
            mask_stack = cv2.drawContours(mask_stack, [c], -1, (255,255,255), -1)

    # refind our final contours
    cnts, h = cv2.findContours(mask_stack[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # visualization of final contours on image
    if visual:
        img = img_np.copy()
        img = cv2.drawContours(img, cnts, -1, (0,255,0), 2)

        cv2.imshow('Image with final contours', img)
        cv2.waitKey(0)

    return cnts, h


def findBoundingBox(mask):
    return None


im_list = []

for i in np.arange(10):
    print(f'processing image: {i}')
    path = f'./data/{CAMERA_ORI}/{SHOE_FACTOR}/color_{i}.jpeg'
    
    img_rgb, img_np = load_img(path)

    mask_b, mask_rgb = segment(img_rgb, img_np, dlab, visual=True)
    cnt, _ = cleanContours(img_np, mask_b, visual=True)

    im_list.append(img_rgb)

# im_list[0].save(f'./results.pdf', save_all=True, append_images=im_list[1:])


