import cv2
import numpy as np
import os


def region_of_interest(img, vertices):
    """创建ROI掩膜"""
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def _auto_canny_thresholds(gray, sigma=0.33):
    """自动计算Canny边缘检测阈值"""
    # 使用中值自适应调整阈值
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # 防止阈值范围过窄
    if upper - lower < 30:
        lower = max(0, lower - 15)
        upper = min(255, upper + 15)
    return lower, upper


def _color_mask_yellow_white(bgr):
    """创建黄色和白色车道线的颜色掩膜"""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    # 白色：高亮度，低饱和度
    # 放宽阈值以适应校园道路（阴影、不同光照条件）
    lower_white = np.array([0, 160, 0], dtype=np.uint8)
    upper_white = np.array([180, 255, 100], dtype=np.uint8)
    mask_white = cv2.inRange(hls, lower_white, upper_white)

    # 黄色：色调约15-40，合适的饱和度和亮度
    lower_yellow = np.array([10, 40, 80], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_white, mask_yellow)

    # 去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _make_roi_vertices(width, height):
    """创建ROI区域（自适应大小的梯形）"""
    bottom_y = height - 1
    top_y = int(height * 0.60)
    # 为校园道路拓宽ROI区域
    left_bottom = int(width * 0.05)
    right_bottom = int(width * 0.95)
    left_top = int(width * 0.35)
    right_top = int(width * 0.65)

    return np.array([[
        (left_bottom, bottom_y),
        (left_top, top_y),
        (right_top, top_y),
        (right_bottom, bottom_y),
    ]], dtype=np.int32)


def _line_slope_intercept(x1, y1, x2, y2):
    """计算直线的斜率和截距"""
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dx) < 1e-6:
        return None
    m = dy / dx
    b = y1 - m * x1
    return m, b


def _filter_and_split_lines(lines, width, height):
    """过滤霍夫直线并分为左右两组"""
    left = []
    right = []
    if lines is None:
        return left, right

    min_len = max(30.0, 0.03 * (width + height))
    min_abs_slope = 0.3  # 放宽斜率限制
    max_abs_slope = 5.0
    center_x = width / 2.0

    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        si = _line_slope_intercept(x1, y1, x2, y2)
        if si is None:
            continue
        m, b = si
        if not (min_abs_slope <= abs(m) <= max_abs_slope):
            continue

        # 偏好ROI下部区域（减少地平线杂波）
        if max(y1, y2) < int(height * 0.55):
            continue

        # 根据斜率和位置分配
        x_mid = 0.5 * (x1 + x2)

        # 存储原始线段点
        item = [x1, y1, x2, y2]

        if m < 0 and x_mid < center_x:
            left.append(item)
        elif m > 0 and x_mid > center_x:
            right.append(item)

    return left, right


def _fit_lane_line(lines, y_min, y_max):
    """使用鲁棒拟合将线段拟合为直线"""
    if lines is None or len(lines) == 0:
        return None

    # 收集所有点
    points = []
    for line in lines:
        x1, y1, x2, y2 = line
        points.append([x1, y1])
        points.append([x2, y2])

    points = np.array(points, dtype=np.int32)

    if len(points) < 2:
        return None

    # 使用鲁棒方法（DIST_HUBER）拟合直线以忽略离群点
    try:
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)

        # 避免水平线除零错误
        if abs(vy) < 1e-6:
            return None

        m_inv = vx / vy

        x1 = int(x0 + (y_max - y0) * m_inv)
        x2 = int(x0 + (y_min - y0) * m_inv)

        return (x1, int(y_max), x2, int(y_min))
    except Exception:
        return None


def _draw_lane_lines(img, left_line, right_line, color=(0, 255, 0), thickness=8):
    """在图像上绘制车道线"""
    overlay = img.copy()
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(img, 0.75, overlay, 1.0, 0.0)


def process_image(image_path, save_debug=False):
    """主处理函数"""
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在。")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法打开图像 {image_path}")
        return

    height, width = image.shape[:2]

    # 1) 偏好车道线颜色（白色/黄色）以抑制背景边缘
    color_mask = _color_mask_yellow_white(image)

    # 2) 直接在颜色掩膜上进行边缘检测
    edges = cv2.Canny(color_mask, 50, 150)

    # 稍微膨胀边缘以填补检测间隙
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel_edge, iterations=1)

    # 3) ROI处理
    roi_vertices = _make_roi_vertices(width, height)
    edges_roi = region_of_interest(edges, roi_vertices)

    # 4) 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(max(20, 0.03 * width)),
        maxLineGap=int(max(100, 0.08 * width)),
    )

    # 备用方案：如果未检测到直线，使用灰度Canny
    if lines is None:
        print("使用颜色掩膜未检测到直线，尝试备用方案（灰度Canny）...")
        gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_raw = cv2.GaussianBlur(gray_raw, (5, 5), 0)
        lower, upper = _auto_canny_thresholds(gray_raw)
        edges = cv2.Canny(gray_raw, lower, upper)
        edges_roi = region_of_interest(edges, roi_vertices)
        lines = cv2.HoughLinesP(
            edges_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(max(20, 0.04 * width)),
            maxLineGap=int(max(50, 0.03 * width)),
        )

    # 5) 过滤、分割和拟合直线
    left_params, right_params = _filter_and_split_lines(lines, width, height)

    y_max = height - 1
    y_min = int(height * 0.60)
    left_line = _fit_lane_line(left_params, y_min=y_min, y_max=y_max)
    right_line = _fit_lane_line(right_params, y_min=y_min, y_max=y_max)

    # 绘制绿色车道线
    output = _draw_lane_lines(image, left_line, right_line)

    output_dir = os.path.dirname(image_path)
    output_path = os.path.join(output_dir, 'result.jpg')
    cv2.imwrite(output_path, output)
    print(f"结果已保存到 {output_path}")

    if save_debug:
        cv2.imwrite(os.path.join(output_dir, 'debug_color_mask.png'), color_mask)
        cv2.imwrite(os.path.join(output_dir, 'debug_edges.png'), edges)
        cv2.imwrite(os.path.join(output_dir, 'debug_edges_roi.png'), edges_roi)

    cv2.imshow('车道线检测', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'picture.jpg')
    process_image(image_path, save_debug=True)