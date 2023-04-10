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
import torch
import torchvision.transforms as T 
from PIL import Image

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


def distance_measure(depth_img: npt.NDArray[np.uint8]) -> int:
    """
    Given depth image, return the minimum depth. 

    Args:
        depth_img (npt.NDArray[np.uint8]): depth image in numpy array format. 

    Returns:
        int: distance in mm.
    """
    nonzero_ind = depth_img[np.where(depth_img > 0)]
    min_dist = np.min(nonzero_ind)

    _logger.info(f"minimum distance found: {min_dist/1000} mm")

    return min_dist


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


def cleanContours(img_np: npt.NDArray[np.uint8], mask_b: npt.NDArray[np.uint8], min_area=1000, 
                  epsilon=0.001, cnt_approx=True, visual=False,):
    """
    Given a mask, detect contours, filter out unwanted contours, and clean up.

    Args:
        img_np (npt.NDArray[np.uint8]): numpy array of original image (H, W, 3).
        mask_b (npt.NDArray[np.unint8]): binary mask with values of 0 or 1, with 0 defining background (H, W, 1).
        min_area (int, optional): minimum area of contour accepted. Defaults to 1000.
        epsilon (int, optional): maximum distance from contour to approximated contour. Defaults to 0.001.
        cnt_approx (bool, optional): approximate the contour with Douglas-Peucker alg. Defaults to True.
        visual (bool, optional): visualize the final contours. Defaults to False.

    Returns:
        npt.NDArray[np.uint8]: binary mask of final contour.
        npt.NDArray[np.uint8]: original image with the contour drawn on top.
    """

    img = img_np.copy()

    # find only the external contours in the mask
    cnts, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of Contours found = " + str(len(cnts)))

    # stacked mask to comply for RGB image dimensions: (N,M,3)
    mask_stack = np.dstack((mask_b,mask_b,mask_b))*255
    mask_result = np.zeros(mask_stack.shape)

    # iterate through each contour, and remove those that are less than a specified size
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            mask_stack = cv2.drawContours(mask_stack, [c], -1, (0,0,0), -1)
        else:
            if cnt_approx:
                eps = epsilon*cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c,eps,True)
                img = cv2.drawContours(img, [approx], -1, (0,255,0), 2)
                mask_result = cv2.drawContours(mask_result, [approx], -1, (255,255,255), -1)
            else:
                img = cv2.drawContours(img, [c], -1, (0,255,0), 2)
                mask_result = cv2.drawContours(mask_result, [c], -1, (255,255,255), -1)

    mask_b = np.sum(mask_result, axis=2).astype(np.uint8)
    mask_b[mask_b > 0] = 1

    # visualization of final contours on image
    if visual:
        cv2.imshow('Image with final contours', img)
        cv2.imshow('final mask', mask_result)
        cv2.waitKey(0)

    return mask_b, img


def findBoundingBox(img_np, mask_b):
    cnts, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbs = np.zeros((1,4,1))
    img = img_np.copy()

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        bbs = np.append(bbs,np.asarray([x,y,w,h]).reshape((1,4,1)),axis=0)

    bbs = bbs[1:,:,:]

    for i in range(bbs.shape[0]):
        bb = bbs[i,:,:].reshape((1,4)).astype(int)
        bb = bb[0]
        img = cv2.rectangle(img,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),(0,0,255),2)

    return bbs,img