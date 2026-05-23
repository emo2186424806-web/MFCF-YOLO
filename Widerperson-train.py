from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")

    model.train(
        data=r"WiderPerson.yaml",
        epochs=10,                                 #100轮
        batch=8,
        imgsz=800,
        workers=8,
        cache=False,
    )