from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 训练模型
train_results = model.train(
    data="/root/LuHY/yolo11/ultralytics-main/ultralytics/cfg/datasets/msrs.yaml",  # 数据集 YAML 路径
    epochs=50, # 300,  # 训练轮次
    imgsz=640,  # 训练图像尺寸
    device="0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
)


# 评估模型在验证集上的性能
metrics = model.val()

# 在图像上执行对象检测
# results = model("path/to/image.jpg")
# results[0].show()

# 将模型导出为 ONNX 格式
path = model.export(format="onnx")  # 返回导出模型的路径

##########################################################################################

# from ultralytics import YOLO

# # 加载模型
# model = YOLO("yolo11n.pt")
# # model = YOLO("yolo11m.pt")

# # 训练模型
# train_results = model.train(
#     data="coco.yaml",  # 数据集 YAML 路径
#     epochs=50, # 300,  # 训练轮次
#     imgsz=640,  # 训练图像尺寸
#     device="0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
# )

# # 评估模型在验证集上的性能
# metrics = model.val()

# # 在图像上执行对象检测
# # results = model("path/to/image.jpg")
# # results[0].show()

# # 将模型导出为 ONNX 格式
# path = model.export(format="onnx")  # 返回导出模型的路径


##########################################################################################

# 导出 ONNX
 
# from ultralytics import YOLO
 
# # 加载一个模型，路径为 YOLO 模型的 .pt 文件
# model = YOLO(r"/root/LuHY/yolo11/ultralytics-main/yolo11n.pt")
# # model = YOLO("/root/LuHY/yolo11/ultralytics-main/yolo11m.pt")
# # model = YOLO("/root/LuHY/yolo11/ultralytics-main/yolo11x.pt")

 
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


##########################################################################################


# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(
#     data="VOC2007.yaml", 
#     epochs=100, 
#     imgsz=640,
#     device=0,
# )

# # 评估模型在验证集上的性能
# metrics = model.val()

# # # 在图像上执行对象检测
# # results = model("path/to/image.jpg")
# # results[0].show()

# # 将模型导出为 ONNX 格式
# path = model.export(format="onnx")  # 返回导出模型的路径


##########################################################################################

# from ultralytics import YOLO

# # Load a pretrained model
# model = YOLO("yolo11n.pt")

# # Train the model on your custom dataset
# model.train(data="VOC.yaml", epochs=100, imgsz=640, device=0)


# # # 将模型导出为 ONNX 格式
# # path = model.export(format="onnx")  # 返回导出模型的路径




# nieve:
# # path: ../datasets/coco-pose # dataset root dir
# train: path/to/datasets/handpose_v2_yolov8_pose/train/images
# val: path/to/datasets/handpose_v2_yolov8_pose/val/images
# # test: test-dev2017.txt # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794
 
# # Keypoints
# kpt_shape: [21, 2] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
# flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13 ,14, 15, 16, 17, 18, 19, 20]
 
# # Add negative image ---------------------------------------------------------------------------------------------------
# negative_setting:
#   neg_ratio: 0.1    # 小于等于0时，按原始官方配置训练，大于0时，控制正负样本。
#   use_extra_neg: True
#   extra_neg_sources: {"path/to/datasets/COCO2017/det_neg/images" : 80000,
#                       # "path/to/datasets/VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages": 1000,
#                       }  # 数据集字符串或列表（图片路径或图片列表）
#   fix_dataset_length: 640000  # 是否自定义每轮参与训练的图片数量
 
# # Classes
# nc: 1
 
# names:
#   0: hand
 
 

# nieve:

# train: path/to/datasets/handpose_v2_yolov8_pose/train/images
# val: path/to/datasets/handpose_v2_yolov8_pose/val/images
