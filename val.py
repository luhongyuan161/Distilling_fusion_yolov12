import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点
# 最终论文的参数量和计算量统一以这个脚本运行出来的为准

if __name__ == '__main__':
    model = YOLO("yolo11x.pt")
    # model = YOLO("./runs/detect/train30/weights/best.pt")
    model.val(data='/root/LuHY/yolo11/ultralytics-main/ultralytics/cfg/datasets/msrs.yaml',
              split='train', # IR
            #   split='val',   # VIS
            #   split='test',  # fusion
              imgsz=640,
              batch=9,
              save_json=True, # if you need to cal coco metrice
              project='msrs',
              name='Fusion',
              )
    
    model.val(data='/root/LuHY/yolo11/ultralytics-main/ultralytics/cfg/datasets/msrs.yaml',
            #   split='train', # IR
              split='val',   # VIS
            #   split='test',  # fusion
              imgsz=640,
              batch=9,
              save_json=True, # if you need to cal coco metrice
              project='msrs',
              name='Fusion',
              )
    
    model.val(data='/root/LuHY/yolo11/ultralytics-main/ultralytics/cfg/datasets/msrs.yaml',
            #   split='train', # IR
            #   split='val',   # VIS
              split='test',  # fusion
              imgsz=640,
              batch=9,
              save_json=True, # if you need to cal coco metrice
              project='msrs',
              name='Fusion',
              )
    
    