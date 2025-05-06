import pyrealsense2 as rs
import numpy as np
import cv2
import random
from ultralytics import YOLO
from norfair import Detection, Tracker

# ————— CONFIG —————
MODEL_PATH      = "yolov8/weights/yolov8n.pt"
CONF_THRESH     = 0.45
DIST_THRESH     = 30    # Norfair matching threshold (px)

# ————— LOAD YOLO & COLORS —————
model      = YOLO(MODEL_PATH)
names_dict = model.names
class_list = [names_dict[i] for i in range(len(names_dict))]
colors     = [
    (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    for _ in class_list
]

# ————— NORFAIR TRACKER —————
tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=DIST_THRESH
)

# ————— RealSense INIT —————
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)
align = rs.align(rs.stream.color)

# ————— WINDOW SETUP —————
# single fullscreen window
WIN_NAME = "Full Frame"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ————— DEPTH ON CLICK —————
aligned_depth = None
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and aligned_depth:
        d = aligned_depth.get_distance(x, y)
        print(f"Depth @({x},{y}) = {d:.3f} m")
cv2.setMouseCallback(WIN_NAME, on_click)

while True:
    try:
        # — capture & align —
        frames        = pipeline.wait_for_frames()
        aligned       = align.process(frames)
        aligned_depth = aligned.get_depth_frame()
        color_frame   = aligned.get_color_frame()
        if not aligned_depth or not color_frame:
            continue

        # — to numpy —
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth.get_data())

        # — YOLO detection each frame —
        results = model.predict(source=[color_image], conf=CONF_THRESH, save=False)
        boxes   = results[0].boxes

        # — Norfair detections —
        detections = []
        for box in boxes:
            clsID = int(box.cls.numpy()[0])
            conf  = float(box.conf.numpy()[0])
            x1,y1,x2,y2 = box.xyxy.numpy()[0]
            pts    = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            scores = np.array([conf]*4)
            det    = Detection(points=pts, scores=scores)
            det.metadata = {'clsID': clsID}
            detections.append(det)

        # — update tracker & draw —
        tracked = tracker.update(detections=detections)
        for obj in tracked:
            pts  = obj.estimate
            x1,y1 = pts.min(axis=0).astype(int)
            x2,y2 = pts.max(axis=0).astype(int)
            clsID = obj.last_detection.metadata['clsID']
            tid   = obj.id

            # depth at box center
            cx, cy = int(pts[:,0].mean()), int(pts[:,1].mean())
            depth_m = aligned_depth.get_distance(cx, cy)

            col = colors[clsID]
            cv2.rectangle(color_image, (x1,y1), (x2,y2), col, 2)
            label = f"{class_list[clsID]} ID:{tid} {depth_m:.2f} m"
            cv2.putText(color_image, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # — depth colormap —
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # — combine & show full frame —
        full_frame = np.hstack((color_image, depth_colormap))
        cv2.imshow(WIN_NAME, full_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)

pipeline.stop()
cv2.destroyAllWindows()
