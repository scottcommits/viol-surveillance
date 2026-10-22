
import cv2
import json

VIDEO = "VIRAT_S_010204_05_000856_000890.mp4"

cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Video frame could not be read")

points = []
window = "Select Grass Boundary"

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window, mouse_callback)

while True:
    display = frame.copy()

    for point in points:
        cv2.circle(display, tuple(point), 5, (0, 0, 255), -1)

    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(
                display,
                tuple(points[i - 1]),
                tuple(points[i]),
                (0, 255, 0),
                2
            )

    if len(points) > 2:
        cv2.line(
            display, tuple(points[-1]), tuple(points[0]),
            (255, 255, 0), 1
        )

    cv2.putText(
        display,
        "Left click: add point | Right click: undo | "
        "Enter: save | R: reset | Q: quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(window, display)
    key = cv2.waitKey(20) & 0xFF

    if key in (10, 13):  # Enter
        if len(points) >= 3:
            with open("grass_polygon.json", "w") as f:
                json.dump(points, f)
            print("Grass coordinates:", points)
            print("Saved to grass_polygon.json")
            break

    elif key == ord("r"):
        points.clear()

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
