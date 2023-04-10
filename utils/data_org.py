import numpy as np
import cv2
from PIL import Image
from pathlib import Path

def numpy_to_jpeg(path, n):
    for i in np.arange(n):
        color_image = np.load(f"{path}/color_{i}.npy")
        cv2.imwrite(f"{path}/color_{i}.jpeg", color_image)


def load_img(path, ori):
    parent_dir = Path().resolve()
    img_rgb = Image.open(parent_dir/path)

    if ori == 'vertical':
        img_rgb = img_rgb.rotate(90, Image.NEAREST, expand = 1)
    img_np = np.array(img_rgb.convert('RGB'))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_rgb, img_np