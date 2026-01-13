import torch
import torchvision
import torchvision.transforms.functional as F
import cv2
import numpy as np
import os

def load_model(device):
    """
    加载在COCO数据集上预训练的Faster R-CNN模型
    """
    print("正在加载模型...")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    print("模型加载成功 (使用FasterRCNN_ResNet50_FPN_Weights)")

    model.to(device)
    model.eval()
    return model

def main():
    # 1. 设置设备
    device = torch.device('cpu')

    # 2. 加载模型
    model = load_model(device)

    # 3. 读取图片
    img_filename = 'picture.jpg'
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_filename)

    if not os.path.exists(img_path):
        print(f"错误：在当前目录下未找到 {img_filename}")
        return

    print(f"正在读取图片: {img_path}")
    # OpenCV读取图片 (BGR格式)
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print("错误：图片读取失败")
        return

    # 转为RGB格式用于模型输入
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # 转为Tensor并归一化 [0, 1]
    img_tensor = F.to_tensor(img_rgb).to(device)
    # 添加Batch维度 [C, H, W] -> [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0)

    # 4. 模型推理
    print("正在进行目标检测...")
    with torch.no_grad():
        prediction = model(img_tensor)

    # 5. 解析结果
    # COCO数据集的类别名称（仅保留必要的类别映射）
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # 设定置信度阈值
    threshold = 0.5

    # 筛选出自行车 (bicycle)
    target_class = 'bicycle'
    detected_indices = [
        i for i, label in enumerate(labels)
        if COCO_INSTANCE_CATEGORY_NAMES[label] == target_class and scores[i] > threshold
    ]

    print(f"检测完成。共检测到 {len(detected_indices)} 辆自行车 (置信度阈值: {threshold})")

    # 6. 绘制结果
    result_img = img_cv.copy()

    for i in detected_indices:
        box = boxes[i]
        score = scores[i]
        x1, y1, x2, y2 = box.astype(int)

        # 打印位置信息
        print(f"自行车位置: [x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}], 置信度: {score:.4f}")

        # 画矩形框 (绿色)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加标签（避免文字超出图片边界）
        text = f"{target_class}: {score:.2f}"
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(result_img, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 7. 保存结果
    output_filename = 'result.jpg'
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    cv2.imwrite(output_path, result_img)
    print(f"检测结果图片已保存至: {output_path}")
    print("实验结束。")

if __name__ == "__main__":
    main()