import numpy as np
import torch
import cv2
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


# model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.show()

test1 = read_image(str(Path('data/horizontal/shoe') / 'color_0.jpeg'))

boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
colors = ["blue", "yellow"]
result = draw_bounding_boxes(test1, boxes, colors=colors, width=5)

show(result)