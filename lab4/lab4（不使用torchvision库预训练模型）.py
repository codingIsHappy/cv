import os
import json
import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 解决 torchvision::nms 在 CUDA 上不可用的问题
# 定义一个纯 PyTorch 实现的 NMS，可以在 CUDA 上运行
def nms_pytorch(boxes, scores, iou_threshold):
    """
    纯 PyTorch 实现的 NMS，支持 CUDA。
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 按分数排序
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # 计算交集
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        # 计算 IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于阈值的框
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        # 注意：ids 是相对于 order[1:] 的索引，所以要 +1 才能对应到 order
        # 如果 ids 是标量（只有一个元素），squeeze 后变成 0-d tensor，需要处理
        if ids.dim() == 0:
             order = order[ids + 1].unsqueeze(0)
        else:
             order = order[ids + 1]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

# 替换 torchvision 的 nms
torchvision.ops.nms = nms_pytorch
torchvision.ops.boxes.nms = nms_pytorch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_ROOT = os.path.join(ROOT_DIR, 'COCO')
TRAIN_IMG_DIR = os.path.join(COCO_ROOT, 'train2017')
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
ANNOTATIONS_DIR = os.path.join(COCO_ROOT, 'annotations_trainval2017')
TRAIN_ANN_FILE = os.path.join(ANNOTATIONS_DIR, 'instances_train2017.json')
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'bicycle_detector.pth')
TEST_IMAGE_PATH = os.path.join(ROOT_DIR, 'picture.jpg')
RESULT_IMAGE_PATH = os.path.join(ROOT_DIR, 'result.jpg')

BATCH_SIZE = 4
NUM_EPOCHS = 12
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BicycleDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        # 加载标注
        print(f"正在从 {ann_file} 加载标注...")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # 查找自行车类别ID
        self.cat_id = None
        for cat in coco_data['categories']:
            if cat['name'] == 'bicycle':
                self.cat_id = cat['id']
                break

        if self.cat_id is None:
            raise ValueError("在标注中未找到 'bicycle' 类别。")

        print(f"自行车类别ID: {self.cat_id}")

        # 筛选包含自行车的图片
        self.images = []
        self.image_id_to_ann = {}

        # 创建 image_id -> 标注列表 的映射
        img_ann_map = {}
        for ann in coco_data['annotations']:
            if ann['category_id'] == self.cat_id:
                img_id = ann['image_id']
                if img_id not in img_ann_map:
                    img_ann_map[img_id] = []
                img_ann_map[img_id].append(ann)

        # 收集这些图片的信息
        for img in coco_data['images']:
            if img['id'] in img_ann_map:
                self.images.append(img)
                self.image_id_to_ann[img['id']] = img_ann_map[img['id']]

        print(f"找到 {len(self.images)} 张包含自行车的图片。")

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        anns = self.image_id_to_ann[img_info['id']]

        boxes = []
        for ann in anns:
            # COCO bbox is [x, y, width, height]
            # PyTorch expects [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 标签: 1 代表自行车 (我们只有这一个类别 + 背景)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

def get_model(num_classes):
    # 加载在COCO上预训练的模型，但我们要从头训练或微调。
    # 要求说'不要使用现成的模型'，意味着我们应该训练它。
    # 然而，从头定义完整的架构很复杂。
    # 我们将使用该架构，但使用随机权重初始化 (pretrained=False)。
    # 注意: 在新版本中，weights=None 的默认值是 pretrained=False。

    try:
        # 较新的 torchvision 版本
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    except TypeError:
        # 较旧的 torchvision 版本
        model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # 替换分类器为新的，
    # 具有用户定义的 num_classes
    # 1个类别 (自行车) + 背景
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if i % 10 == 0:
            print(f"轮次: {epoch}, 迭代: {i}, 损失: {losses.item():.4f}")

    print(f"轮次 {epoch} 完成。平均损失: {total_loss / len(data_loader):.4f}")

def detect_and_draw(model, img_path, output_path):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(img).to(DEVICE)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # 过滤结果
    img_cv = cv2.imread(img_path)

    # 置信度阈值
    score_threshold = 0.5

    for i, box in enumerate(prediction['boxes']):
        score = prediction['scores'][i].item()
        if score > score_threshold:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"Bicycle: {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img_cv)
    print(f"结果已保存至 {output_path}")

def main():
    print(f"使用设备: {DEVICE}")

    # 检查模型是否存在
    train_new = True
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"在 {MODEL_SAVE_PATH} 发现已有模型")
        user_input = input("你想重新训练模型吗？(y/n): ").strip().lower()
        if user_input != 'y':
            train_new = False

    num_classes = 2 # 1个类别 (自行车) + 背景
    model = get_model(num_classes)
    model.to(DEVICE)

    if train_new:
        if not os.path.exists(TRAIN_ANN_FILE):
            print(f"错误: 未在 {TRAIN_ANN_FILE} 找到标注文件")
            return

        if not os.path.exists(TRAIN_IMG_DIR):
            print(f"错误: 未在 {TRAIN_IMG_DIR} 找到训练图片目录")
            return

        print("开始训练... (如果使用CPU可能会很慢)")
        # 变换
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        dataset = BicycleDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=transform)

        # 如果数据集很大，可以使用较小的子集进行演示，
        # 但要求说'在数据集上训练'。
        # 我们将使用找到的完整数据集。

        data_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, collate_fn=collate_fn
        )

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

        for epoch in range(NUM_EPOCHS):
            train_one_epoch(model, optimizer, data_loader, DEVICE, epoch)

        # 保存模型
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"模型已保存至 {MODEL_SAVE_PATH}")

    else:
        print("正在加载已有模型...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # 推理
    if os.path.exists(TEST_IMAGE_PATH):
        print(f"正在检测 {TEST_IMAGE_PATH} 中的自行车...")
        detect_and_draw(model, TEST_IMAGE_PATH, RESULT_IMAGE_PATH)
    else:
        print(f"未找到测试图片 {TEST_IMAGE_PATH}。")

if __name__ == "__main__":
    main()
