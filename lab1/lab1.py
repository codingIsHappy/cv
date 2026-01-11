import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def manual_convolution(image, kernel):
    """
    手动执行2D卷积，使用numpy切片以提高效率。
    """
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # 用零填充图像
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output = np.zeros_like(image, dtype=np.float32)
    
    # 向量化滑动窗口
    for i in range(k_h):
        for j in range(k_w):
            # 移动图像并乘以核权重
            # 我们正在累加加权和
            output += padded_image[i:i+h, j:j+w] * kernel[i, j]
            
    return output

def sobel_operator(image):
    """
    应用Sobel算子和给定的特定核。
    返回:
        sobel_magnitude: Gx和Gy的组合幅值
        given_kernel_output: 说明书中提供的特定核的输出
    """
    # 实验指导PPT中给定的核（垂直边缘/水平梯度）
    # 1  0 -1
    # 2  0 -2
    # 1  0 -1
    Gx_kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)
    
    # Gy核（水平边缘/垂直梯度）
    Gy_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)
    
    # 应用卷积
    gx = manual_convolution(image, Gx_kernel)
    gy = manual_convolution(image, Gy_kernel)
    
    # 计算幅值
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # 归一化到0-255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    # “给定核输出”本质上是gx（通常取绝对值或平移，但让我们保持原始或绝对值）
    # 通常为了可视化，我们取绝对值并截断
    given_output = np.abs(gx)
    given_output = np.clip(given_output, 0, 255).astype(np.uint8)
    
    return magnitude, given_output

def manual_histogram(image):
    """
    手动计算颜色直方图。
    """
    h, w, c = image.shape
    
    # 初始化B、G、R的直方图
    hist_b = np.zeros(256, dtype=int)
    hist_g = np.zeros(256, dtype=int)
    hist_r = np.zeros(256, dtype=int)
    
    # 遍历所有像素（在纯python中可能很慢，在这里尝试稍微高效一点）
    # 纯手动循环：
    # for i in range(h):
    #     for j in range(w):
    #         b, g, r = image[i, j]
    #         hist_b[b] += 1
    #         hist_g[g] += 1
    #         hist_r[r] += 1
            
    # 使用扁平数组的稍微优化的手动方法，但仍然是手动计数逻辑
    flat_b = image[:, :, 0].flatten()
    flat_g = image[:, :, 1].flatten()
    flat_r = image[:, :, 2].flatten()
    
    for val in flat_b:
        hist_b[val] += 1
    for val in flat_g:
        hist_g[val] += 1
    for val in flat_r:
        hist_r[val] += 1
        
    return hist_b, hist_g, hist_r

def plot_histogram(hist_b, hist_g, hist_r):
    plt.figure(figsize=(10, 5))
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Pixel Count")

    plt.plot(hist_r, color='r', label='Red')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_b, color='b', label='Blue')

    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, alpha=0.3)

def manual_lbp(image):
    """
    手动实现局部二值模式（LBP）。
    """
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    
    # 填充图像以处理边界
    padded = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    
    # 8个邻居的权重（顺时针从左上角开始或类似）
    # 标准LBP：
    # 1 2 4
    # 128 0 8
    # 64 32 16
    weights = np.array([
        [1, 2, 4],
        [128, 0, 8],
        [64, 32, 16]
    ], dtype=np.uint8)
    
    # 迭代
    # 在这里使用循环对于“手动实现”是可以接受的
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            center = padded[i, j]
            code = 0
            
            # 左上
            if padded[i-1, j-1] >= center: code += 1
            # 上
            if padded[i-1, j] >= center: code += 2
            # 右上
            if padded[i-1, j+1] >= center: code += 4
            # 右
            if padded[i, j+1] >= center: code += 8
            # 右下
            if padded[i+1, j+1] >= center: code += 16
            # 下
            if padded[i+1, j] >= center: code += 32
            # 左下
            if padded[i+1, j-1] >= center: code += 64
            # 左
            if padded[i, j-1] >= center: code += 128
            
            output[i-1, j-1] = code
            
    return output

def calculate_texture_features(lbp_image):
    """
    计算LBP图像的直方图作为纹理特征。
    """
    # 使用numpy进行更快的直方图计算
    hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
    
    # 归一化
    hist = hist.astype(float) / (lbp_image.size + 1e-7)
    return hist

def main():
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载图像
    img_path = os.path.join(script_dir, 'picture.jpg')
    if not os.path.exists(img_path):
        print(f"错误：未找到 {img_path}。")
        return

    # 读取为彩色以用于直方图
    img_color = cv2.imread(img_path)
    if img_color is None:
        print("错误：读取图像失败。")
        return
        
    # 读取为灰度以用于滤波和纹理
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    print("正在处理图像...")

    # 2. 滤波（Sobel和给定核）
    print("正在应用滤波器...")
    sobel_mag, given_kernel_res = sobel_operator(img_gray)
    
    plt.figure()
    plt.imshow(sobel_mag, cmap='gray')
    plt.title("Sobel Magnitude")

    plt.figure()
    plt.imshow(given_kernel_res, cmap='gray')
    plt.title("Given Kernel Result")

    # 3. 颜色直方图
    print("正在计算直方图...")
    hist_b, hist_g, hist_r = manual_histogram(img_color)
    plot_histogram(hist_b, hist_g, hist_r)

    # 4. 纹理特征（LBP）
    print("正在提取纹理特征...")
    lbp_img = manual_lbp(img_gray)
    # cv2.imwrite(os.path.join(script_dir, 'output_lbp_vis.jpg'), lbp_img) # 可选的可视化

    plt.figure()
    plt.imshow(lbp_img, cmap='gray')
    plt.title("LBP Visualization")

    texture_features = calculate_texture_features(lbp_img)
    np.save(os.path.join(script_dir, 'texture_features.npy'), texture_features)
    
    print("完成！结果已展示。")
    plt.show()

if __name__ == "__main__":
    main()