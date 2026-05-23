from ultralytics import YOLO

# 加载 YOLO11n 模型
model = YOLO("yolo11n.pt")

# ===================== 正确：评估性能（用官方 coco128 数据集） =====================
metrics = model.val(
    data="coco128.yaml",  # 官方自带数据集，必须用这个！
    imgsz=640,
    batch=8,
    device=0,
    verbose=True,
)

# 获取速度、参数
speed = model.speed
params, gflops = model.info()

# 输出完整性能指标
print("\n" + "=" * 65)
print("             YOLO11n 目标检测性能指标")
print("=" * 65)
print(f"mAP50-95    : {metrics.box.map:.4f}")
print(f"mAP50       : {metrics.box.map50:.4f}")
print(f"Precision   : {metrics.box.p:.4f}")
print(f"Recall      : {metrics.box.r:.4f}")

fps = 1000 / speed["inference"]
print(f"\nSpeed 推理  : {speed['inference']:.2f} ms/im")
print(f"FPS         : {fps:.1f}")
print(f"Params      : {params:.2f} M")
print(f"GFLOPs      : {gflops:.2f}")
print("=" * 65)
