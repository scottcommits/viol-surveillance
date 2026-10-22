from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo26n.pt")

# Test on an image
results = model.predict(
    source="test.jpg",
    save=True
)

print("Detection complete!")