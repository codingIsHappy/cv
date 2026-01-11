import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import cv2
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\n测试集: 平均损失: {test_loss:.4f}, '
          f'准确率: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

def predict_student_id(model, device, image_path):
    if not os.path.exists(image_path):
        print(f"未找到图片 {image_path}。跳过预测。")
        return

    print(f"正在处理图片: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("加载图片失败。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化：反转二值（文字变白，背景变黑）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积过滤轮廓（去除噪声）
    # 此阈值可能需要根据图像分辨率进行调整
    min_area = 20
    digit_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # 从左到右排序轮廓
    digit_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

    predicted_id = ""

    model.eval()
    for c in digit_contours:
        x, y, w, h = cv2.boundingRect(c)

        # 提取感兴趣区域 (ROI)
        roi = thresh[y:y+h, x:x+w]

        # 调整大小为 28x28 并填充以保持纵横比
        # 我们希望数字适应 28x28 中心 20x20 的框
        target_size = 20
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            continue

        resized_roi = cv2.resize(roi, (new_w, new_h))

        # 放置在 28x28 画布的中心
        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_roi

        # 归一化
        # ToTensor 将 [0, 255] 转换为 [0.0, 1.0]
        tensor_img = transforms.ToTensor()(canvas)
        tensor_img = transforms.Normalize((0.1307,), (0.3081,))(tensor_img)

        tensor_img = tensor_img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            pred = output.argmax(dim=1, keepdim=True)
            predicted_id += str(pred.item())

    print(f"预测学号: {predicted_id}")
    return predicted_id

def main():
    # 训练设置
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 1.0
    gamma = 0.7

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 数据预处理与增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机正负十度旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),   # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准归一化
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 检查本地是否存在 MNIST 数据
    data_root = './data'
    mnist_exists = os.path.exists(os.path.join(data_root, 'MNIST'))

    if mnist_exists:
        print("检测到本地 MNIST 数据集。")
        download_flag = False
    else:
        print("未找到本地 MNIST 数据集。正在下载...")
        download_flag = True

    try:
        dataset1 = datasets.MNIST(data_root, train=True, download=download_flag, transform=train_transform)
        dataset2 = datasets.MNIST(data_root, train=False, download=download_flag, transform=test_transform)
    except RuntimeError as e:
        print(f"加载数据集出错: {e}")
        print("尝试下载...")
        dataset1 = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        dataset2 = datasets.MNIST(data_root, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    save_path = "mnist_cnn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")

    # 预测学号
    predict_student_id(model, device, "student_id.png")

if __name__ == '__main__':
    main()