from typing import Tuple

import pyrealsense2 as rs
import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
import time

CAMERA_ORI = "horizontal"
MAX_SAVE_FILES = 5

def grabCut(img: npt.NDArray[np.uint8], rect: Tuple[int, int, int, int], iterations: int = 5) -> npt.NDArray[np.uint8]:
    """
    Provided an approximate bounding box containing the object of interest, 
    performs a GrabCut image segmentation.

    Args:
        img (npt.NDArray[np.uint8]): input image in RGB format
        iterations (int): iterations for GrabCut algorithm
        rect (Tuple(int, int, int, int)): approx. bounding box of object

    Returns:
        npt.NDArray[np.uint8]: mask of foreground object
    """

    mask = np.zeros(img.shape[:2], dtype="uint8")

    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    start = time.time()
    (mask, bgModel, fgModel) = cv2.grabCut(img, mask, rect, bgModel,
        fgModel, iterCount=iterations, mode=cv2.GC_INIT_WITH_RECT)
    end = time.time()
    print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

    values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
    )
    # loop over the possible GrabCut mask values
    # for (name, value) in values:
    #     # construct a mask that for the current value
    #     print("[INFO] showing mask for '{}'".format(name))
    #     valueMask = (mask == value).astype("uint8") * 255
    #     # display the mask so we can visualize it
    #     cv2.imshow(name, valueMask)
    #     cv2.waitKey(0)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    # scale the mask from the range [0, 1] to [0, 255]
    outputMask = (outputMask * 255).astype("uint8")
    # apply a bitwise AND to the image using our mask generated by
    # GrabCut to generate our final output image
    output = cv2.bitwise_and(img, img, mask=outputMask)

    # cv2.imshow("Input", depth_colormap)
    # cv2.imshow("GrabCut Mask", outputMask)
    # cv2.imshow("GrabCut Output", output)
    # cv2.waitKey(0)

    return output

def visualize_img(color_img: npt.NDArray[np.uint8], depth_img: npt.NDArray[np.uint8]):

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_img.shape    

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    plt.show()
    cv2.waitKey(0)


for i in np.arange(5):
    # depth image default scale is one millimeter 
    depth_image = np.load(f"data/{CAMERA_ORI}/shoe/depth_{i}.npy")
    color_image = np.load(f"data/{CAMERA_ORI}/shoe/color_{i}.npy")

    if CAMERA_ORI == "vertical":
        depth_image = cv2.rotate(depth_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        color_image = cv2.rotate(color_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    depth_mask = np.zeros(depth_image.shape)
    depth_mask[np.where(depth_image > 0)] = 1
    cv2.imshow("mask 1", depth_mask)
    depth_mask[np.where(depth_image > 800)] = 0
    cv2.imshow("mask", depth_mask)
    cv2.waitKey(0)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # cv2.imshow('Original', color_image)
    # cv2.waitKey(0)  

    x = np.arange(640)
    plt.plot(x, depth_image[240, :])

    # -------------------------------------------------------------------------
    # GrabCut algorithm
    bounding_box = (127, 0, 371-127, 366)
    masked_img = grabCut(color_image, bounding_box)

    # -------------------------------------------------------------------------
    # # Edge detection
    # img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

    # # Sobel Edge Detection
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) 
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv2.waitKey(0)

    # # Canny Edge Detection
    # edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.waitKey(0)
