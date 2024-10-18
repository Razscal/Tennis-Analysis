from ultralytics import YOLO

model = YOLO('yolov8x.pt')

result = model.track('tennis_olympic_2024.mp4', conf=0.2, save=True)

for box in result[0]:
    print(box)