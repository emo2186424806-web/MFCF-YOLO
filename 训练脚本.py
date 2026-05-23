from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")

    model.train(
        data=r"coco8.yaml",
        epochs=10,                                 #10轮
        batch=1,
        imgsz=640,
        workers=0,
        cache=False,
    )