import pyrealsense2 as rs
import numpy as np
import cv2
import random
from ultralytics import YOLO

# Opening the file containing COCO class names
my_file = open("yolov8/utils/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
my_file.close()

# Generate random colors for each class
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load the YOLOv8 model
model = YOLO("yolov8/weights/yolov8n.pt", "v8")

# Vals to resize video frames (optional, can improve performance)
frame_wid = 640
frame_hyt = 480

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Set up streams (depth and color)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
profile = pipeline.start(config)

# Create an alignment object to align depth with color frames
align = rs.align(rs.stream.color)


# Function to handle mouse click event (optional)
def show_distance(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = aligned_depth_frame.get_distance(x, y)
        print(f"Distance at pixel ({x}, {y}): {depth} meters")


# Set up a window for mouse click callback (optional)
cv2.namedWindow('Color Frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Color Frame', show_distance)

# Main streaming loop
try:
    while True:
        # Capture frames from RealSense
        frames = pipeline.wait_for_frames()

        # Align depth frame to color frame
        aligned_frames = align.process(frames)

        # Get the aligned depth and color frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # YOLOv8 detection on the color frame
        detect_params = model.predict(source=[color_image], conf=0.45, save=False)

        # Convert tensor array to numpy (YOLO predictions)
        DP = detect_params[0].numpy()

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]  # Get one detected box
                clsID = int(box.cls.numpy()[0])
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                # Calculate the center of the bounding box
                center_x = int((bb[0] + bb[2]) / 2)
                center_y = int((bb[1] + bb[3]) / 2)

                # Get the depth value at the center of the bounding box
                depth = aligned_depth_frame.get_distance(center_x, center_y)

                # Draw bounding box around the detected object
                cv2.rectangle(
                    color_image,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[clsID],
                    3,
                )

                # Display class name, confidence, and distance at the center of the bounding box
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    color_image,
                    f"{class_list[clsID]}  {round(depth, 3)} metr",#{round(conf, 3):.3f}
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        # Display the resulting frame with YOLO detections and distances
        cv2.imshow("Color Frame", color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    # Release the camera and close any OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
