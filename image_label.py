import numpy as np
import cv2
from torchvision import models
import PIL
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms as T 

from utils.post_processing import filter_depth, approx_bounding_box, perform_grabcut, visualize_boundbox


CAMERA_ORI = "horizontal"
SHOE_FACTOR = "shoe"
MAX_SAVE_FILES = 5

# import models
fcn = models.segmentation.fcn_resnet101(weights=models.segmentation.FCN_ResNet101_Weights.DEFAULT).eval()
dlab = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()


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


def segment(net, path):
    img = PIL.Image.open(path)
    img = img.rotate(90, PIL.Image.NEAREST, expand = 1)
    cv2_img = np.array(img.convert('RGB'))
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    cv2.imshow('original', cv2_img)
    
    # plt.imshow(img); plt.axis('off'); plt.show()

    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([
                    # T.Resize(256), 
                    # T.CenterCrop(224), 
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    cv2.imshow('mask', rgb)
    # plt.imshow(rgb); plt.axis('off'); plt.show()
    cv2.waitKey(0)

for i in np.arange(5):
    segment(dlab, f'./data/vertical/shoe/color_{i}.jpeg')

