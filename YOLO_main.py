import cvzone
from ultralytics import YOLO
import cv2
import numpy as np

# For Videos
cap = cv2.VideoCapture("Videos/footage.mp4")
# Dimensions for resizing
max_width = 900
max_height = 900

if not cap.isOpened():
    print("Video Error")

# Chose the model YOLO-weight needed, if not already available will be downloaded to the path
model = YOLO("YOLO-weights/yolov8m.pt")

fps = cap.get(cv2.CAP_PROP_FPS)

count = 0
counted = {}

CAR_AVG_WIDTH = 1.75
line1_times = {}
line2_times = {}
speed_by_id = {}
LINE_DISTANCE_IN_METERS = 20

width_samples = []
meters_per_pixel = None
prev_positions = {}
prev_centers = {}
speed_history = {}
speed_kmph = 0

frame_number = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_number += 1

    height, width = img.shape[:2]

    if height > width:
        scale = max_height / height
    else:
        scale = max_width / width

    width_new = int(width * scale)
    height_new = int(height * scale)

    img = cv2.resize(img, (width_new, height_new))

    # Counting line
    line_y = int(img.shape[0] * 3 / 4)
    cv2.line(img, (0, line_y), (img.shape[1], line_y), (0, 255, 0), 2)

    # Lines for two line speed calculation
    line_start_y = int(img.shape[0]*1/3)
    line_end_y = int(img.shape[0]*1/2)

    cv2.line(img, (0, line_start_y), (img.shape[1], line_start_y), (0, 0, 255), 2)
    cv2.line(img, (0, line_end_y), (img.shape[1], line_end_y), (255, 0, 0), 2)

    results = model.track(img, persist=True, tracker="botsort.yaml", classes=[2], conf=0.5) #iou == intersection over union, 2 is the class of car

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is None:      # occasionally (first frame, edge cases), BoT-SORT may not have assigned an ID yet.
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w, h = x2 - x1, y2 - y1

            # Center of the bounding box
            cX, cY = int(x1 + w / 2), int(y1 + h / 2)

            track_id = int(box.id[0])

            # Scale calibration near the counting line (for instantaneous speed estimation only)
            if track_id not in speed_by_id:
                if meters_per_pixel is None:
                    if 60 < w < 200 and abs(cY - line_y) < 10:
                        width_samples.append(w)

                    if len(width_samples) >= 20:
                        avg_width = np.median(width_samples)
                        meters_per_pixel = (CAR_AVG_WIDTH / avg_width)

            # Speed Estimation (Instantaneous)
            if meters_per_pixel is not None:
                if track_id in prev_positions:
                    prevX, prevY, prevFrame = prev_positions[track_id]

                    dx = cX - prevX
                    dy = cY - prevY
                    pixel_dist = np.hypot(dx, dy)

                    dt = (frame_number - prevFrame) / fps
                    if dt > 0:
                        speed_mps = (pixel_dist * meters_per_pixel) / dt
                        speed_kmph = speed_mps * (3600/1000)

                    # Reduce fluctuations using a moving average
                    speed_history.setdefault(track_id, []).append(speed_kmph)
                    speed_history[track_id] = speed_history[track_id][-5:]  # keep only the last 5 values
                    speed_kmph = np.average(speed_history[track_id])

                    speed_kmph = max(0, min(speed_kmph, 140))   # Reduces jitter

                prev_positions[track_id] = (cX, cY, frame_number)

            # Speed estimation (Two line method)
            if track_id in prev_centers:
                prevY = prev_centers[track_id]
                # Crossing first line
                if track_id not in line1_times and prevY < line_start_y <= cY:
                    line1_times[track_id] = frame_number

                # Crossing second line
                if track_id in line1_times and track_id not in line2_times and prevY < line_end_y <= cY:
                    line2_times[track_id] = frame_number

                    # Speed calculation
                    dt = (line2_times[track_id] - line1_times[track_id]) / fps
                    if dt > 0:
                        speed_mps = LINE_DISTANCE_IN_METERS / dt
                        speed_by_id[track_id] = round((speed_mps * (3600/1000)), 3) # in kmph

            prev_centers[track_id] = cY

            # Counting
            if track_id not in counted:     # not yet counted
                counted[track_id] = False

            if not counted[track_id] and cY > line_y:
                counted[track_id] = True
                count += 1

            cvzone.cornerRect(img, (x1, y1, w, h), l=15)
            instantaneous_val_colorR = (235,152,116)
            instantaneous_val_colorT = (0,0,0)
            if track_id in speed_by_id:
                speed_kmph = speed_by_id[track_id]
                instantaneous_val_colorR = (0, 255, 0)
                cvzone.putTextRect(img, f'{speed_by_id[track_id]} kmph', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=10, colorR = (0,0,0), colorT=(0,0,255))

            # Instantaneous speed
            #cvzone.putTextRect(img, f'{speed_kmph:.1f}', (max(0, x2), max(35, y2)), scale=1, thickness=1, offset=10, colorR=instantaneous_val_colorR, colorT=instantaneous_val_colorT)
            # ID on the bottom right of the bounding box
            cvzone.putTextRect(img, f'Id : {track_id}', (max(0, x2), max(35, y2)), scale=1, thickness=1, offset=10, colorR=instantaneous_val_colorR, colorT=instantaneous_val_colorT)
            cv2.circle(img, (cX, cY), 5, (123, 255, 24), cv2.FILLED)

        if meters_per_pixel is None:
            mmp = "Calculating"
        else:
            mmp = f'{meters_per_pixel:.3f}'

        cv2.putText(img, f'Count: {count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # cv2.putText(img, f'METERS_PER_PIXEL: {mmp}', (int(img.shape[1]/4), 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Text below line starting and ending measurement
        cv2.putText(img, 'Start ', (10, line_start_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, 'End ', (10, line_end_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, 'Counting Line ', (10, line_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv2.putText(img, "Red: Start | Blue: End | Green: Count",
        #             (10, img.shape[0] - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break