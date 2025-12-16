import argparse
import cv2
import numpy as np
import onnxruntime as ort
import os
from pycocotools.coco import COCO
from tqdm import tqdm

# 类外定义类别映射关系，使用字典格式
CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class YOLO11:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = CLASS_NAMES
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self, input_image):
        self.img = cv2.imread(input_image)
        if self.img is None:
            raise ValueError(f"Image not found at path: {input_image}")
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(640, 640))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # 原图形状
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 新的大小
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算padding
        dw /= 2
        dh /= 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # 调整大小
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 填充边缘
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)  # 强制调整为 640x640
        return img, (r, r), (dw, dh)


    def draw_detections(self, image, box, score, class_id):
        """
        Draws bounding box and label on the image.
        
        :param image: The input image to draw on.
        :param box: The bounding box [left, top, width, height].
        :param score: The confidence score of the prediction.
        :param class_id: The predicted class ID.
        """
        label = f"{self.classes[class_id]}: {score:.2f}"
        color = self.color_palette[class_id]  # Assign a unique color to each class

        # Draw the bounding box
        left, top, width, height = box
        right = left + width
        bottom = top + height

        # Draw rectangle around the object
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

        # Draw label
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Optionally, you can also draw the class label in a background box for better visibility
        cv2.rectangle(image, (left, top - 20), (left + len(label) * 10, top), color, -1)  # Background box
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Text with black color

    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))  # 转置输出
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []  # Initialize the lists here
        ratio = self.img_width / 640, self.img_height / 640  # 比例
        for i in range(rows):
            classes_scores = outputs[i][4:]  # 类别得分
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:  # 置信度过滤
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][:4]
                x -= self.dw
                y -= self.dh
                x /= ratio[0]
                y /= ratio[1]
                w /= ratio[0]
                h /= ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        # 进行 NMS（非最大抑制）
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 由于 NMS 的结果可能是一个二维数组或者是一个嵌套数组
        if len(indices) > 0:
            # 确保 indices 是二维数组形式，如果是标量则转为数组
            indices = indices.flatten() if len(indices.shape) > 1 else indices

            for i in indices:
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                self.draw_detections(input_image, box, score, class_id)
        
        # Return the boxes, scores, and class_ids here as well
        return boxes, scores, class_ids

    def predict(self, input_image):
        # 加载并预处理
        img_data = self.preprocess(input_image)
        # 推理
        session = ort.InferenceSession(self.onnx_model, 
            providers=["CUDAExecutionProvider","CPUExecutionProvider"] if ort.get_device()=="GPU" else ["CPUExecutionProvider"])
        output = session.run(None, {session.get_inputs()[0].name: img_data})
        # 后处理，得到列表
        return self.postprocess(self.img, output)  # Pass both self.img (input_image) and output here



    def main(self, input_image):
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"])
        img_data = self.preprocess(input_image)
        outputs = session.run(None, {session.get_inputs()[0].name: img_data})
        return self.postprocess(self.img, outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="runs/detect/train/weights/best.onnx")
    parser.add_argument("--val_annotations", type=str, default="/root/autodl-tmp/COCO2017/annotations/captions_val2017.json", help="COCO 验证集的 JSON 注释文件路径")
    parser.add_argument("--val_images", type=str, default="/root/autodl-tmp/COCO2017/images/val2017", help="COCO 验证集图像文件夹路径")
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    args = parser.parse_args()

    coco = COCO(args.val_annotations)
    image_ids = coco.getImgIds()
    detector = YOLO11(args.model, args.conf_thres, args.iou_thres)

    results = []  # 存放所有检测结果
    # for img_id in image_ids:
    for img_id in tqdm(image_ids, desc="Processing Images", unit="image"):
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.val_images, info['file_name'])
        boxes, scores, class_ids = detector.predict(img_path)

        for box, score, cid in zip(boxes, scores, class_ids):
            results.append({
                "image_id":    img_id,
                "category_id": cid,
                "bbox":        [round(x,2) for x in box],
                "score":       float(score)
            })
        print(f"[{len(results)} detections] processed {info['file_name']}")

    # 将结果保存为 JSON
    import json
    json_path = "coco_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f)

    # 加载检测结果并评估
    from pycocotools.cocoeval import COCOeval
    coco_pred = coco.loadRes(json_path)
    evaluator = COCOeval(coco, coco_pred, iouType='bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    









# # Ultralytics YOLO 🚀, AGPL-3.0 license
 
# import argparse
# import cv2
# import numpy as np
# import onnxruntime as ort
 
# # 类外定义类别映射关系，使用字典格式
# CLASS_NAMES = {
#     0: 'person',
#     1: 'bicycle',
#     2: 'car',
#     3: 'motorcycle',
#     4: 'airplane',
#     5: 'bus',
#     6: 'train',
#     7: 'truck',
#     8: 'boat',
#     9: 'traffic light',
#     10: 'fire hydrant',
#     11: 'stop sign',
#     12: 'parking meter',
#     13: 'bench',
#     14: 'bird',
#     15: 'cat',
#     16: 'dog',
#     17: 'horse',
#     18: 'sheep',
#     19: 'cow',
#     20: 'elephant',
#     21: 'bear',
#     22: 'zebra',
#     23: 'giraffe',
#     24: 'backpack',
#     25: 'umbrella',
#     26: 'handbag',
#     27: 'tie',
#     28: 'suitcase',
#     29: 'frisbee',
#     30: 'skis',
#     31: 'snowboard',
#     32: 'sports ball',
#     33: 'kite',
#     34: 'baseball bat',
#     35: 'baseball glove',
#     36: 'skateboard',
#     37: 'surfboard',
#     38: 'tennis racket',
#     39: 'bottle',
#     40: 'wine glass',
#     41: 'cup',
#     42: 'fork',
#     43: 'knife',
#     44: 'spoon',
#     45: 'bowl',
#     46: 'banana',
#     47: 'apple',
#     48: 'sandwich',
#     49: 'orange',
#     50: 'broccoli',
#     51: 'carrot',
#     52: 'hot dog',
#     53: 'pizza',
#     54: 'donut',
#     55: 'cake',
#     56: 'chair',
#     57: 'couch',
#     58: 'potted plant',
#     59: 'bed',
#     60: 'dining table',
#     61: 'toilet',
#     62: 'tv',
#     63: 'laptop',
#     64: 'mouse',
#     65: 'remote',
#     66: 'keyboard',
#     67: 'cell phone',
#     68: 'microwave',
#     69: 'oven',
#     70: 'toaster',
#     71: 'sink',
#     72: 'refrigerator',
#     73: 'book',
#     74: 'clock',
#     75: 'vase',
#     76: 'scissors',
#     77: 'teddy bear',
#     78: 'hair drier',
#     79: 'toothbrush'
# }

 
# class YOLO11:
#     """YOLO11 目标检测模型类，用于处理推理和可视化。"""
#     def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
#         """
#         初始化 YOLO11 类的实例。
#         参数：
#             onnx_model: ONNX 模型的路径。
#             input_image: 输入图像的路径。
#             confidence_thres: 用于过滤检测结果的置信度阈值。
#             iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
#         """
#         self.onnx_model = onnx_model
#         self.input_image = input_image
#         self.confidence_thres = confidence_thres
#         self.iou_thres = iou_thres
 
#         # 加载类别名称
#         self.classes = CLASS_NAMES
 
#         # 为每个类别生成一个颜色调色板
#         self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
 
#     def preprocess(self):
#         """
#         对输入图像进行预处理，以便进行推理。
#         返回：
#             image_data: 经过预处理的图像数据，准备进行推理。
#         """
#         # 使用 OpenCV 读取输入图像
#         self.img = cv2.imread(self.input_image)
#         # 获取输入图像的高度和宽度
#         self.img_height, self.img_width = self.img.shape[:2]
 
#         # 将图像颜色空间从 BGR 转换为 RGB
#         img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
 
#         # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
#         img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))
 
#         # 通过除以 255.0 来归一化图像数据
#         image_data = np.array(img) / 255.0
 
#         # 将图像的通道维度移到第一维
#         image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
 
#         # 扩展图像数据的维度，以匹配模型输入的形状
#         image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
 
#         # 返回预处理后的图像数据
#         return image_data
 
#     def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
#         """
#         将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
#         """
#         shape = img.shape[:2]  # 当前图像的宽高
#         print(f"Original image shape: {shape}")
 
#         if isinstance(new_shape, int):
#             new_shape = (new_shape, new_shape)
 
#         # 计算缩放比例
#         r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
#         if not scaleup:  # 仅缩小，不放大
#             r = min(r, 1.0)
 
#         # 缩放后的未填充尺寸
#         new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
 
#         # 计算需要的填充
#         dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
#         dw /= 2  # padding 均分
#         dh /= 2
 
#         # 缩放图像
#         if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
#             img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
 
#         # 为图像添加边框以达到目标尺寸
#         top, bottom = int(round(dh)), int(round(dh))
#         left, right = int(round(dw)), int(round(dw))
#         img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#         print(f"Final letterboxed image shape: {img.shape}")
 
#         return img, (r, r), (dw, dh)
 
#     def postprocess(self, input_image, output):
#         """
#         对模型输出进行后处理，以提取边界框、分数和类别 ID。
#         参数：
#             input_image (numpy.ndarray): 输入图像。
#             output (numpy.ndarray): 模型的输出。
#         返回：
#             numpy.ndarray: 包含检测结果的输入图像。
#         """
#         # 转置并压缩输出，以匹配预期形状
#         outputs = np.transpose(np.squeeze(output[0]))
#         rows = outputs.shape[0]
#         boxes, scores, class_ids = [], [], []
 
#         # 计算缩放比例和填充
#         ratio = self.img_width / self.input_width, self.img_height / self.input_height
 
#         for i in range(rows):
#             classes_scores = outputs[i][4:]
#             max_score = np.amax(classes_scores)
#             if max_score >= self.confidence_thres:
#                 class_id = np.argmax(classes_scores)
#                 x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
 
#                 # 将框调整到原始图像尺寸，考虑缩放和填充
#                 x -= self.dw  # 移除填充
#                 y -= self.dh
#                 x /= self.ratio[0]  # 缩放回原图
#                 y /= self.ratio[1]
#                 w /= self.ratio[0]
#                 h /= self.ratio[1]
#                 left = int(x - w / 2)
#                 top = int(y - h / 2)
#                 width = int(w)
#                 height = int(h)
 
#                 boxes.append([left, top, width, height])
#                 scores.append(max_score)
#                 class_ids.append(class_id)
 
#         indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
#         for i in indices:
#             box = boxes[i]
#             score = scores[i]
#             class_id = class_ids[i]
#             self.draw_detections(input_image, box, score, class_id)
#         return input_image

#     def draw_detections(self, img, box, score, class_id):
#         """
#         在输入图像上绘制检测到的边界框和标签。
#         参数：
#             img: 用于绘制检测结果的输入图像。
#             box: 检测到的边界框。
#             score: 对应的检测分数。
#             class_id: 检测到的目标类别 ID。      
#         返回：
#             None
#         """
#         # 提取边界框的坐标
#         x1, y1, w, h = box
 
#         # 获取类别对应的颜色
#         color = self.color_palette[class_id]
 
#         # 在图像上绘制边界框
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
 
#         # 创建包含类别名和分数的标签文本
#         label = f"{self.classes[class_id]}: {score:.2f}"
 
#         # 计算标签文本的尺寸
#         (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
 
#         # 计算标签文本的位置
#         label_x = x1
#         label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
 
#         # 绘制填充的矩形作为标签文本的背景
#         cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
 
#         # 在图像上绘制标签文本
#         cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
 
#     def main(self):
#         # 使用 ONNX 模型创建推理会话，自动选择CPU或GPU
#         session = ort.InferenceSession(
#             self.onnx_model, 
#             providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],
#         )
#         # 打印模型的输入尺寸
#         print("YOLO11 🚀 目标检测 ONNXRuntime")
#         print("模型名称：", self.onnx_model)
        
#         # 获取模型的输入形状
#         model_inputs = session.get_inputs()
#         input_shape = model_inputs[0].shape  
#         self.input_width = input_shape[2]
#         self.input_height = input_shape[3]
#         print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height}")
 
#         # 预处理图像数据，确保使用模型要求的尺寸 (640x640)
#         img_data = self.preprocess()
 
#         # 使用预处理后的图像数据运行推理
#         outputs = session.run(None, {model_inputs[0].name: img_data})
 
#         # 对输出进行后处理以获取输出图像
#         return self.postprocess(self.img, outputs)  # 输出图像
 
# if __name__ == "__main__":
#     # 创建参数解析器以处理命令行参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.onnx", help="输入你的 ONNX 模型路径。")
#     parser.add_argument("--img", type=str, default=r"/root/autodl-tmp/COCO2017/images/test/000000000139.jpg", help="输入图像的路径。")
#     parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
#     parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
#     args = parser.parse_args()
 
#     # 使用指定的参数创建 YOLO11 类的实例
#     detection = YOLO11(args.model, args.img, args.conf_thres, args.iou_thres)
 
#     # 执行目标检测并获取输出图像
#     output_image = detection.main()
 
#     # 保存输出图像到文件
#     cv2.imwrite("det_result_picture.jpg", output_image)
 
#     print("图像已保存为 det_result_picture.jpg")
 