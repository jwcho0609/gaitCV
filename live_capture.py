import os

import pyrealsense2 as rs
import numpy as np
import cv2
import PIL


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

CAMERA_ORI = "horizontal"
SHOE_FACTOR = "shoe"
CAPTURE_FREQ = 10

# calculate the number of images currently in the data folder
path = f"./data/{CAMERA_ORI}/{SHOE_FACTOR}"
N = int((len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])) / 3)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device.hardware_reset()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# Start streaming
profile = pipeline.start(config)

image_capture = False
frame_count = 1

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # if not depth_frame or not color_frame:
        #     continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_jpg = color_frame.get_data()
        color_image = np.asanyarray(color_jpg)

        # color_image = increase_brightness(color_image, value=20)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(
                color_image,
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA,
            )
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # # Show images
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)
        key = cv2.waitKey(1)

        if image_capture and frame_count == CAPTURE_FREQ:
            print(f"image captured at: color_{N}")
            np.save(f"./data/{CAMERA_ORI}/{SHOE_FACTOR}/depth_{N}.npy", depth_image)
            np.save(f"./data/{CAMERA_ORI}/{SHOE_FACTOR}/color_{N}.npy", color_image)
            cv2.imwrite(f"./data/{CAMERA_ORI}/{SHOE_FACTOR}/color_{N}.jpeg", color_image)

            N += 1

        if key == ord("q"):
            break
        elif key == ord("c"):
            image_capture = not image_capture
            print(f"capture images: {image_capture}")

        frame_count += 1
        if frame_count > CAPTURE_FREQ:
            frame_count = 1

finally:
    # Stop streaming
    pipeline.stop()
