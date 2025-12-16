import argparse
import cv2
import numpy as np
from openvino.runtime import Core
import time
import os

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
    def __init__(self, xml_model, input_image, confidence_thres, iou_thres):
        self.xml_model = xml_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = CLASS_NAMES
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # OpenVINO 初始化
        self.ie = Core()
        self.model = self.ie.read_model(model=self.xml_model)
        self.compiled_model = self.ie.compile_model(model=self.model, device_name="CPU")
        
        # 获取输入输出信息（新API方式）
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # 获取输入尺寸
        self.input_height, self.input_width = self.input_layer.shape[2], self.input_layer.shape[3]

    def preprocess(self):
        self.img = cv2.imread(self.input_image)
        if self.img is None:
            raise ValueError(f"无法加载图像: {self.input_image}")
            
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        # Letterbox处理
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, (self.input_width, self.input_height))
        
        # 归一化并转换维度
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC to CHW
        return np.expand_dims(image_data, axis=0).astype(np.float32)  # Add batch dim

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """保持纵横比的resize填充"""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
            
        # 计算比例
        r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
        new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
        dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
        dw, dh = dw/2, dh/2
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # 填充
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def postprocess(self, input_image, output):
        """处理输出并绘制检测框"""
        outputs = np.squeeze(output[0])  # 去除batch维度
        boxes, scores, class_ids = [], [], []
        
        # 假设输出形状为[84, 8400]
        for i in range(outputs.shape[1]):
            classes_scores = outputs[4:, i]
            max_score = np.max(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[:4, i]
                
                # 坐标转换
                x = (x - self.dw) / self.ratio[0]
                y = (y - self.dh) / self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                
                boxes.append([x-w/2, y-h/2, w, h])
                scores.append(max_score)
                class_ids.append(class_id)

                # 输出每个预测框的坐标
                print(f"检测框 {i}: 类别: {self.classes[class_id]}, 坐标: {x-w/2}, {y-h/2}, {w}, {h}, 置信度: {max_score:.2f}")
        
        # NMS过滤
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        
        # 检查返回格式
        if isinstance(indices, list):
            indices = [i[0] for i in indices]  # 处理返回的是列表的情况
        elif isinstance(indices, np.ndarray):
            indices = indices.flatten()  # 转为一维数组
        
        # 绘制结果
        for i in indices:
            box = boxes[i]
            self.draw_detections(input_image, box, scores[i], class_ids[i])
            
        return input_image


    def draw_detections(self, img, box, score, class_id):
        """绘制单个检测结果"""
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        
        # 画框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, 2)
        
        # 标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 标签背景
        label_x, label_y = int(x1), int(y1 - 10 if y1 - 10 > label_height else y1 + 10)
        cv2.rectangle(img, 
                     (label_x, label_y - label_height),
                     (label_x + label_width, label_y + label_height),
                     color, cv2.FILLED)
        
        # 标签文字
        cv2.putText(img, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    def main(self):
        """主流程"""
        img_data = self.preprocess()
        
        # 新版OpenVINO推理方式
        output = self.compiled_model([img_data])[self.output_layer]
        
        result_img = self.postprocess(self.img.copy(), output)
        cv2.imwrite("openvino_16.jpg", result_img)
        # print("检测结果已保存为 detection_result.jpg")
        return result_img

# 文件夹图
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="/root/LuHY/yolo11/ultralytics-main/yolo11n_16.xml", help="OpenVINO模型路径")
    parser.add_argument("--model", type=str, default="/root/LuHY/yolo11/ultralytics-main/yolo11n_32.xml", help="OpenVINO模型路径")
    parser.add_argument("--img_folder", type=str, default="/root/LuHY/yolo11/ultralytics-main/images/", help="输入图像文件夹路径")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--save_folder", type=str, default="./result", help="结果保存文件夹")
    args = parser.parse_args()
    print("使用 OpenVINO IR 批量推理...")

    # 创建结果文件夹（如果不存在）
    os.makedirs(args.save_folder, exist_ok=True)

    # 获取所有图像文件
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in os.listdir(args.img_folder) if os.path.splitext(f)[-1].lower() in img_extensions]

    print(f"检测到 {len(img_files)} 张图片，开始处理...")

    total_time = 0.0

    for img_file in img_files:
        img_path = os.path.join(args.img_folder, img_file)
        # print(f"正在处理: {img_path}")

        # 定义保存路径
        save_path = os.path.join(args.save_folder, img_file)

        # 初始化并处理图像
        detector = YOLO11(args.model, img_path, args.conf_thres, args.iou_thres)

        start = time.time()
        # 注意：这里假设 detector.main() 返回的是处理后的图像（如OpenCV格式）
        output_image = detector.main()
        end = time.time()

        # 保存图像（使用cv2）
        import cv2
        cv2.imwrite(save_path, output_image)

        # 记录耗时
        inference_time = (end - start) * 1000
        print(f"{img_file} 推理耗时: {inference_time:.2f} ms")

        total_time += inference_time

    avg_time = (total_time / len(img_files))   # 毫秒

    print("--------------------------------------------")
    print(f"共处理 {len(img_files)} 张图片")
    print(f"平均每张图片处理时间 = {avg_time:.2f} ms")
    print(f"结果已保存至: {args.save_folder}")
    print("--------------------------------------------")




















# import argparse
# import cv2
# import numpy as np
# from openvino.runtime import Core
# import time
# import os

# # 类外定义类别映射关系，使用字典格式
# CLASS_NAMES = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
#     6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
#     11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
#     16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
#     22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
#     27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
#     32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
#     36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
#     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
#     45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
#     50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
#     55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
#     60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
#     65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
#     70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
#     75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'

# }

# class YOLO11:
#     def __init__(self, xml_model, input_image, confidence_thres, iou_thres):
#         self.xml_model = xml_model
#         self.input_image = input_image
#         self.confidence_thres = confidence_thres
#         self.iou_thres = iou_thres
#         self.classes = CLASS_NAMES
#         self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
#         # OpenVINO 初始化
#         self.ie = Core()
#         self.model = self.ie.read_model(model=self.xml_model)
#         self.compiled_model = self.ie.compile_model(model=self.model, device_name="CPU")
        
#         # 获取输入输出信息（新API方式）
#         self.input_layer = self.compiled_model.input(0)
#         self.output_layer = self.compiled_model.output(0)
        
#         # 获取输入尺寸
#         self.input_height, self.input_width = self.input_layer.shape[2], self.input_layer.shape[3]

#     def preprocess(self):
#         self.img = cv2.imread(self.input_image)
#         if self.img is None:
#             raise ValueError(f"无法加载图像: {self.input_image}")
            
#         self.img_height, self.img_width = self.img.shape[:2]
#         img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
#         # Letterbox处理
#         img, self.ratio, (self.dw, self.dh) = self.letterbox(img, (self.input_width, self.input_height))
        
#         # 归一化并转换维度
#         image_data = np.array(img) / 255.0
#         image_data = np.transpose(image_data, (2, 0, 1))  # HWC to CHW
#         return np.expand_dims(image_data, axis=0).astype(np.float32)  # Add batch dim

#     def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
#         """保持纵横比的resize填充"""
#         shape = img.shape[:2]
#         if isinstance(new_shape, int):
#             new_shape = (new_shape, new_shape)
            
#         # 计算比例
#         r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
#         new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
#         dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
#         dw, dh = dw/2, dh/2
        
#         # Resize
#         if shape[::-1] != new_unpad:
#             img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
#         # 填充
#         top, bottom = int(round(dh)), int(round(dh))
#         left, right = int(round(dw)), int(round(dw))
#         img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#         return img, (r, r), (dw, dh)

#     def postprocess(self, input_image, output):
#         """处理输出并绘制检测框"""
#         outputs = np.squeeze(output[0])  # 去除batch维度
#         boxes, scores, class_ids = [], [], []
        
#         # 假设输出形状为[84, 8400]
#         for i in range(outputs.shape[1]):
#             classes_scores = outputs[4:, i]
#             max_score = np.max(classes_scores)
#             if max_score >= self.confidence_thres:
#                 class_id = np.argmax(classes_scores)
#                 x, y, w, h = outputs[:4, i]
                
#                 # 坐标转换
#                 x = (x - self.dw) / self.ratio[0]
#                 y = (y - self.dh) / self.ratio[1]
#                 w /= self.ratio[0]
#                 h /= self.ratio[1]
                
#                 boxes.append([x-w/2, y-h/2, w, h])
#                 scores.append(max_score)
#                 class_ids.append(class_id)
        
#         # NMS过滤
#         indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        
#         # 检查返回格式
#         if isinstance(indices, list):
#             indices = [i[0] for i in indices]  # 处理返回的是列表的情况
#         elif isinstance(indices, np.ndarray):
#             indices = indices.flatten()  # 转为一维数组
        
#         # 绘制结果
#         for i in indices:
#             box = boxes[i]
#             self.draw_detections(input_image, box, scores[i], class_ids[i])
            
#         return input_image


#     def draw_detections(self, img, box, score, class_id):
#         """绘制单个检测结果"""
#         x1, y1, w, h = box
#         color = self.color_palette[class_id]
        
#         # 画框
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, 2)
        
#         # 标签文本
#         label = f"{self.classes[class_id]}: {score:.2f}"
#         (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
#         # 标签背景
#         label_x, label_y = int(x1), int(y1 - 10 if y1 - 10 > label_height else y1 + 10)
#         cv2.rectangle(img, 
#                      (label_x, label_y - label_height),
#                      (label_x + label_width, label_y + label_height),
#                      color, cv2.FILLED)
        
#         # 标签文字
#         cv2.putText(img, label, (label_x, label_y), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

#     def main(self):
#         """主流程"""
#         img_data = self.preprocess()
        
#         # 新版OpenVINO推理方式
#         output = self.compiled_model([img_data])[self.output_layer]
        
#         result_img = self.postprocess(self.img.copy(), output)
#         cv2.imwrite("openvino_16.jpg", result_img)
#         # print("检测结果已保存为 detection_result.jpg")
#         return result_img

# # 单图
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--model", type=str, default="/root/LuHY/yolo11/ultralytics-main/yolo11n_16.xml", help="OpenVINO模型路径")
# #     parser.add_argument("--img", type=str, default="/root/LuHY/yolo11/ultralytics-main/images/000000000139.jpg", help="输入图像路径")
# #     parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
# #     parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
# #     args = parser.parse_args()

# #     start = time.time()
# #     detector = YOLO11(args.model, args.img, args.conf_thres, args.iou_thres)
# #     end = time.time()
# #     print("--------------------------------------------")
# #     print("total_time = ", (end - start) * 1000, 'ms')
# #     print("--------------------------------------------")
# #     detector.main()


# # 文件夹图
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="/root/LuHY/yolo11/ultralytics-main/yolo11n_16.xml", help="OpenVINO模型路径")
#     # parser.add_argument("--model", type=str, default="/root/LuHY/yolo11/ultralytics-main/yolo11n_32.xml", help="OpenVINO模型路径")
#     parser.add_argument("--img_folder", type=str, default="/root/LuHY/yolo11/ultralytics-main/images/", help="输入图像文件夹路径")
#     parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
#     parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
#     parser.add_argument("--save_folder", type=str, default="./result", help="结果保存文件夹")
#     args = parser.parse_args()
#     print("使用 OpenVINO IR 批量推理...")

#     # 创建结果文件夹（如果不存在）
#     os.makedirs(args.save_folder, exist_ok=True)

#     # 获取所有图像文件
#     img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
#     img_files = [f for f in os.listdir(args.img_folder) if os.path.splitext(f)[-1].lower() in img_extensions]

#     print(f"检测到 {len(img_files)} 张图片，开始处理...")

#     total_time = 0.0

#     for img_file in img_files:
#         img_path = os.path.join(args.img_folder, img_file)
#         # print(f"正在处理: {img_path}")

#         # 定义保存路径
#         save_path = os.path.join(args.save_folder, img_file)

#         # 初始化并处理图像
#         detector = YOLO11(args.model, img_path, args.conf_thres, args.iou_thres)

#         start = time.time()
#         # 注意：这里假设 detector.main() 返回的是处理后的图像（如OpenCV格式）
#         output_image = detector.main()
#         end = time.time()

#         # 保存图像（使用cv2）
#         import cv2
#         cv2.imwrite(save_path, output_image)

#         # 记录耗时
#         inference_time = (end - start) * 1000
#         print(f"{img_file} 推理耗时: {inference_time:.2f} ms")

#         total_time += inference_time

#     avg_time = (total_time / len(img_files))   # 毫秒

#     print("--------------------------------------------")
#     print(f"共处理 {len(img_files)} 张图片")
#     print(f"平均每张图片处理时间 = {avg_time:.2f} ms")
#     print(f"结果已保存至: {args.save_folder}")
#     print("--------------------------------------------")




