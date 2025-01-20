import sys
import numpy as np

def caculate_brightness(img):
    if img.mode == 'RGB':  # 彩色图像
        # 将图像转换为 numpy 数组并计算每个通道的平均亮度
        np_image = np.array(img)
        r = np_image[:, :, 0]
        g = np_image[:, :, 1]
        b = np_image[:, :, 2]
        avg_brightness = (np.mean(r) + np.mean(g) + np.mean(b)) / 3
    elif img.mode == 'L':  # 灰度图像
        np_image = np.array(img)
        avg_brightness = np.mean(np_image)
    return avg_brightness


def calculate_image_contrast(img):
    image = img.convert('L')  # 转换为灰度图像
    np_image = np.array(image)
    min_val = np.min(np_image)
    max_val = np.max(np_image)
    contrast = (max_val - min_val) / (max_val + min_val)
    return contrast