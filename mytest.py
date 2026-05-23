import torch

from ultralytics import YOLO


# 计算 FLOPs 工具函数（直接嵌入，不用额外装库）
def calculate_flops(model, imgsz=640):
    try:
        # YOLO 自带方法，最稳定、不报错
        model.model.info(verbose=False)
        # 获取参数量和计算量
        params = sum(x.numel() for x in model.model.parameters())
        # 自动获取 GFLOPs（ultralytics 内置支持）
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        from thop import profile

        flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
        return params, flops
    except:
        return 0, 0


if __name__ == "__main__":
    # 加载模型
    model = YOLO("runs/detect/train2/weights/best.pt")

    # ===================== 打印模型信息 =====================
    print("\n" + "=" * 65)
    print("            模型结构信息")
    print("=" * 65)
    model.info()

    # ===================== 计算 参数 + GFLOPs =====================
    params, flops = calculate_flops(model, imgsz=640)
    params_M = params / 1e6
    gflops = flops / 1e9 if flops != 0 else 0

    # ===================== 验证模型 =====================
    metrics = model.val(data="WiderPerson.yaml", split="test", imgsz=640, batch=8, workers=8, verbose=True)

    # ===================== 输出指标（可直接写论文） =====================
    print("\n" + "=" * 65)
    print("            模型指标汇总（可直接写论文）")
    print("=" * 65)
    print(f"📌 模型参数 (Params):    {params_M:.2f} M")
    print(f"📌 模型计算量 (GFLOPs):   {gflops:.2f} G")  # <-- 这里就是新增的计算量
    print(f"🎯 精确率 (Precision):   {metrics.box.mp:.2%}")
    print(f"🎯 召回率 (Recall):      {metrics.box.mr:.2%}")
    print(f"🎯 mAP@0.5:              {metrics.box.map50:.2%}")
    print(f"🎯 mAP@0.5:0.95:         {metrics.box.map:.2%}")
    print("=" * 65)
