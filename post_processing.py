import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt

CAMERA_ORI = "vertical"
MAX_SAVE_FILES = 5

for i in np.arange(5):
    # depth image default scale is one millimeter 
    depth_image = np.load(f"data/{CAMERA_ORI}/shoe/depth_{i}.npy")
    color_image = np.load(f"data/{CAMERA_ORI}/shoe/color_{i}.npy")

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    cv2.imshow('Original', color_image)
    cv2.waitKey(0)  

    x = np.arange(640)
    plt.plot(x, depth_image[240, :])

    # -------------------------------------------------------------------------
    # GrabCut
    mask = np.zeros(color_image.shape[:2], dtype="uint8")

    # dec_filter = rs.decimation_filter()

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





    ## Visualization
    # If depth and color resolutions are different, resize color image to match depth image for display
    # if depth_colormap_dim != color_colormap_dim:
    #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
    #     images = np.hstack((resized_color_image, depth_colormap))
    # else:
    #     images = np.hstack((color_image, depth_colormap))

    # # Show images
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', depth_colormap)
    # plt.show()
    # key = cv2.waitKey(0)
