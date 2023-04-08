from typing import Tuple
import logging

from pathlib import Path
import time

import pyrealsense2 as rs
import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
import skimage

_logger = logging.getLogger(__name__)


def filter_depth(
    depth_img: npt.NDArray[np.uint8], ori: str, shoe_factor: str, lower_bound: int = 0, upper_bound: int = 800
):
    """
    Filter out the depth image for values outside of the provided lower and upper bounds.

    Args:
        depth_img (npt.NDArray[np.uint8]): depth image
        lower_bound (int): lower distance value to filter out of depth (mm)
        upper_bound (int): upper distance value to filter out of depth (mm)
    """

    _logger.info(
        f"filtering depth based on lower and upper bounds: ({lower_bound}, {upper_bound}) mm"
    )

    # empty mask declaration
    depth_mask = np.zeros(depth_img.shape, dtype=np.uint8)

    # filter for the lower bound
    depth_mask[np.where(depth_img > lower_bound)] = 255
    first_mask = depth_mask.copy()

    # filter for the upper bound
    depth_mask[np.where(depth_img > upper_bound)] = 0

    images = np.hstack((first_mask, depth_mask))

    # save depth image
    curr_path = Path(__file__).parent
    save_path = curr_path / f"test/{ori}_{shoe_factor}_depth_mask.jpg"

    cv2.imwrite(str(save_path), images)

    # cv2.imshow("filtered depth mask", images)
    # cv2.waitKey(0)

    return depth_mask


def approx_bounding_box(
    color_image: npt.NDArray[np.uint8], depth_mask: npt.NDArray[np.uint8], buffer: int, ori: str, clean: bool = True
) -> Tuple[int, int, int, int]:
    """
    Approximate the bounding box using mathematical approach.

    Args:
        depth_mask (npt.NDArray[np.uint8]): depth mask of image
        buffer (int): buffer for the right/left edge of leg

    Returns:
        Tuple[int, int, int, int]: bounding box (x, y, w, h)
    """

    _logger.info("Approximating the bounding box.")

    dim = depth_mask.shape

    # find the left edge of the bounding box
    # maximum vertical pixels to consider [0, 1] with 1 being 100% of height from top of image
    vert_bound = 0.5
    mask_bounded = depth_mask[0 : int(depth_mask.shape[0] * vert_bound), :]
    left_bound = 0

    # sweep columns from left to right and stop when a column with pixel is found (left edge of leg)
    for i in np.arange(dim[1]):
        col = mask_bounded[:, i]

        if any(col):
            left_bound = i
            break

    # _logger.info(f"Left edge: {left_bound}")

    # find the right edge of the bounding box
    # apply closing morph transform (dilate-erode) to clean noise
    kernel = np.ones((10, 10), np.uint8)
    cleaned_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)

    # images = np.hstack((depth_mask, cleaned_mask))
    # cv2.imshow("filter", images)
    # cv2.waitKey(0)

    x_val = []
    y_val = []

    for i in np.arange(dim[1]):
        # iterate from the right edge of the image
        col = cleaned_mask[:, dim[1] - i - 1]

        # find row of where mask exists
        first_ind = (np.where(col > 0)[0])[0]
        y_val.insert(0, first_ind)
        x_val.insert(0, dim[1] - i - 1)

        # if we reach the right edge of the leg
        if not first_ind:
            break

    # find the maximum peak of the y values, aka where the corner is
    peak_ind = (np.where(y_val == max(y_val))[0])[-1]
    right_bound = x_val[peak_ind]

    # _logger.info(f"Right edge: {right_bound}")

    if ori == "horizontal":
        y_buffer = 25
    else:
        y_buffer = 100

    (x, y, w, h) = (left_bound, 0, right_bound - left_bound + buffer, dim[0] - y_buffer)

    if clean:
        bounding_box = (x, y, w, h)
        grabcut_mask = perform_grabcut(color_image, bounding_box)
        y_max = np.max(np.nonzero(np.sum(grabcut_mask, axis=1)))
        h = y_max

    _logger.info(f"Bounding box: ({x}, {y}, {w}, {h})")

    return (x, y, w, h)


def perform_grabcut(
    color_img: npt.NDArray[np.uint8],
    rect: Tuple[int, int, int, int],
    iterations: int = 5,
) -> npt.NDArray[np.uint8]:
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

    _logger.info(f"Performing GrabCut algorithm with {iterations} iterations.")

    mask = np.zeros(color_img.shape[:2], dtype="uint8")

    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    start = time.time()
    (mask, bgModel, fgModel) = cv2.grabCut(
        color_img,
        mask,
        rect,
        bgModel,
        fgModel,
        iterCount=iterations,
        mode=cv2.GC_INIT_WITH_RECT,
    )
    end = time.time()
    _logger.info("Applying GrabCut took {:.2f} seconds".format(end - start))

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

    # kernel = skimage.morphology.disk(8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    (thresh, binRed) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

    # mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    
    # scale the mask from the range [0, 1] to [0, 255]
    outputMask = (outputMask * 255).astype("uint8")

    return outputMask


def visualize_img(
    color_img: npt.NDArray[np.uint8],
    depth_img: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8] = None,
    bound_box: Tuple[int, int, int, int] = None,
):
    """
    Visualize the color and depth image side by side. The depth image has color map applied.

    Args:
        color_img (npt.NDArray[np.uint8]): color image
        depth_img (npt.NDArray[np.uint8]): depth image
        mask (npt.NDArray[np.uint8]): mask for the leg
    """

    # apply the mask to the depth image and set outliers to 2000 mm away
    if mask is not None:
        for r in np.arange(mask.shape[0] - 1):
            for c in np.arange(mask.shape[1] - 1):
                if not mask[r, c]:
                    depth_img[r, c] = 2000

    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
    )

    # draw bounding box on color image
    if bound_box is not None:
        color_copy = color_img.copy()
        x = bound_box[0]
        y = bound_box[1]
        w = bound_box[2]
        h = bound_box[3]
        cv2.rectangle(color_copy, (x, y), (x + w, y + h), (255, 0, 0), 4)

    # cv2.imshow("bounded_box", color_image)
    # cv2.waitKey(0)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_img.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(
            color_img,
            dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
            interpolation=cv2.INTER_AREA,
        )
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        if bound_box is not None:
            images = np.hstack((color_copy, depth_colormap))
        else:
            images = np.hstack((color_img, depth_colormap))

    if mask is not None:
        color_crop = color_img.copy()
        color_crop = cv2.bitwise_and(color_crop, color_crop, mask=mask)
        images = np.hstack((images, color_crop))

    # plot a straight line across the image at some y value
    # x = np.arange(640)
    # plt.plot(x, depth_image[240, :])
    # plt.show()

    # Show images
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", images)
    cv2.waitKey(0)


def visualize_boundbox(color_img: npt.NDArray[np.uint8], bound_box: Tuple[int, int, int, int]):
    # draw bounding box on color image
    color_copy = color_img.copy()
    x = bound_box[0]
    y = bound_box[1]
    w = bound_box[2]
    h = bound_box[3]
    cv2.rectangle(color_copy, (x, y), (x + w, y + h), (255, 0, 0), 4)

    cv2.imshow("bounded_box", color_copy)
    cv2.waitKey(0)


def distance_measure(depth_img: npt.NDArray[np.uint8]):
    nonzero_ind = depth_img[np.where(depth_img > 0)]
    min_dist = np.min(nonzero_ind)

    _logger.info(f"minimum distance found: {min_dist/1000} mm")

    return min_dist
