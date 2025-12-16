# test_msrs.py
from ultralytics import YOLO

def main():
    # ===== 路径设置 =====
    weights_path = "yolo11n.pt"        # 训练好的模型权重
    data_yaml = "/root/LuHY/yolo11/ultralytics-main/ultralytics/cfg/datasets/msrs.yaml" # 数据集配置文件
    
    # ===== 加载模型 =====
    model = YOLO(weights_path)
    
    # ===== 在测试集上评估 =====
    results = model.val(
        data=data_yaml,
        split="test",   # 指定使用测试集
        imgsz=640,     # 输入尺寸
        batch=16,      # batch size
        device=0       # 使用GPU 0，改成 "cpu" 也行
    )
    
    # ===== 输出结果 =====
    print("评估完成 ✅")
    print(results)  # 包含 mAP, Precision, Recall 等指标
    
    # ===== 单张图片测试示例 =====
    test_image = "/root/autodl-fs/MSRS-main/detection/ir/1.png"
    preds = model.predict(source=test_image, save=True)
    print("单张推理完成，结果保存在 runs/detect/predict/")

if __name__ == "__main__":
    main()
