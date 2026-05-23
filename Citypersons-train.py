from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11.yaml")

    model.train(
        data=r"CityPersons.yaml",
        epochs=50,  # 100轮
        batch=16,
        imgsz=640,
        workers=8,
        cache=False,
    )
