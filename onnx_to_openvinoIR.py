import openvino as ov

# 导出模型为 ONNX 格式
onnx_path = '/root/LuHY/yolo11/ultralytics-main/yolo11m_16.onnx'

# 转换 ONNX 模型为 OpenVINO 格式
ov_model = ov.convert_model(onnx_path)

# 保存 OpenVINO 模型
ir_path = '/root/LuHY/yolo11/ultralytics-main/yolo11m_16.xml'
ov.save_model(ov_model, ir_path)

print("OpenVINO IR model saved to:", ir_path)






# # 导出 ONNX
 
# from ultralytics import YOLO
 
# # 加载一个模型，路径为 YOLO 模型的 .pt 文件
# model = YOLO(r"/root/LuHY/yolo11/ultralytics-main/yolo11n.pt")
 
# # 导出模型，设置多种参数
# model.export(
#     format="onnx",      # 导出格式为 ONNX
#     imgsz=(640, 640),   # 设置输入图像的尺寸
#     keras=False,        # 不导出为 Keras 格式
#     optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
#     half=False,         # 不启用 FP16 量化
#     int8=True,         # 不启用 INT8 量化
#     dynamic=True,      # 不启用动态输入尺寸
#     simplify=False,      # 简化 ONNX 模型
#     opset=None,         # 使用最新的 opset 版本
#     workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
#     nms=False,          # 不添加 NMS（非极大值抑制）
#     batch=1,            # 指定批处理大小
#     device="cpu"        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
# )
