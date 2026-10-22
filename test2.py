from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.predict(
    source="VIRAT_S_010204_05_000856_000890.mp4",
    save=True,
    conf=0.35,
    vid_stride=2
)

print("Video analysis complete!")
