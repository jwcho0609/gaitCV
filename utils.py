import numpy as np
import cv2

def numpy_to_jpeg(path, n):
    for i in np.arange(n):
        color_image = np.load(f"{path}/color_{i}.npy")
        cv2.imwrite(f"{path}/color_{i}.jpeg", color_image)