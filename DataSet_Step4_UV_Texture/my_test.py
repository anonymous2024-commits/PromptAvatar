import cv2
import numpy as np

# 读取输入图像
img = cv2.imread('template_tex_match_color.jpg')

# 定义重映射数组
map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # 计算目标位置
        map_x[i,j] = i
        map_y[i,j] = j

# 进行重映射
dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# 保存输出图像
cv2.imwrite('output.jpg', dst)