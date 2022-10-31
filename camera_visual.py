import pyrealsense2 as rs
import numpy as np
import cv2

MAX_SAVE_FILES = 5

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

curr_save_count = 0

print("Press 'c' to capture and 'q' to quit.")

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # pixel_distance_in_meters = depth_frame.get_distance()

        dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
        spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
        temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
        filtered = dec_filter.process(depth_frame)
        filtered = spat_filter.process(filtered)
        filtered = temp_filter.process(filtered)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('c'):
            np.save(f"vertical/no_shoe/depth_{curr_save_count}.npy", depth_image)
            np.save(f"vertical/no_shoe/color_{curr_save_count}.npy", color_image)
            print(f'image {curr_save_count} captured')
            
            curr_save_count += 1
            if curr_save_count > MAX_SAVE_FILES - 1:
                curr_save_count = 0

finally:

    # Stop streaming
    pipeline.stop()