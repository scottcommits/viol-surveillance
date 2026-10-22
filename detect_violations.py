
import cv2
import json
import numpy as np
from ultralytics import YOLO

VIDEO = "VIRAT_S_010204_05_000856_000890.mp4"
MODEL = "yolo26n.pt"

with open("grass_polygon.json", "r") as f:
    points = json.load(f)

# Remove duplicate points
points = list(dict.fromkeys(tuple(p) for p in points))

if len(points) < 3:
    raise ValueError("Grass polygon needs at least 3 unique points.")

grass_polygon = np.array(points, dtype=np.int32)

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

# Number of consecutive processed frames in grass
REQUIRED_FRAMES = 8
grass_frames = {}
reported_ids = set()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    result = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )[0]

    # Draw configured grass zone
    overlay = frame.copy()
    cv2.fillPoly(overlay, [grass_polygon], (0, 255, 0))
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.polylines(frame, [grass_polygon], True, (0, 255, 0), 2)

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.int().cpu().tolist()

        for track_id, (x1, y1, x2, y2) in zip(ids, boxes):
            # Approximate ground contact point
            foot = ((x1 + x2) // 2, y2)

            inside = cv2.pointPolygonTest(
                grass_polygon,
                (float(foot[0]), float(foot[1])),
                False
            ) >= 0

            grass_frames[track_id] = (
                grass_frames.get(track_id, 0) + 1
                if inside else 0
            )

            violation = grass_frames[track_id] >= REQUIRED_FRAMES
            color = (0, 0, 255) if violation else (255, 200, 0)

            label = f"Person {track_id}"
            if violation:
                label += " - CHECK VIOLATION"

                if track_id not in reported_ids:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    print(
                        f"Potential grass violation: ID={track_id}, "
                        f"time={timestamp:.2f}s"
                    )
                    reported_ids.add(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    cv2.imshow("Grass Zone Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
