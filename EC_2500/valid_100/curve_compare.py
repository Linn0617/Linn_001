import os
import numpy as np
import matplotlib.pyplot as plt

# 创建保存图像的文件夹
os.makedirs('curves', exist_ok=True)

# 读取pred文件
with open('pred.txt', 'r') as f:
    pred_lines = f.readlines()

# 处理每条曲线
for i in range(1, 101):
    # 读取原始曲线数据
    curve_file = os.path.join('curve', f'{i}.txt')
    curve_data = np.loadtxt(curve_file)

    # 提取荷载(y)和位移(x)
    load = curve_data[:, 0]  # 第一列为荷载
    displacement = -1000 * curve_data[:, 1]  # 第二列的-1000倍为位移

    # 解析pred中的对应行
    pred_points = np.array([float(x) for x in pred_lines[i - 1].split()])

    # 创建绘图
    plt.figure(figsize=(10, 6))

    # 绘制原始曲线
    plt.plot(displacement, load, 'b-', linewidth=1.5, label='True')

    # 绘制预测折线（连接所有4个点）
    if len(pred_points) >= 4:
        # 提取四个坐标点 (x1, y1, x2, y2, x3, y3, x4, y4)
        points = []
        for j in range(0, len(pred_points), 2):
            if j + 1 < len(pred_points):
                points.append((pred_points[j], pred_points[j + 1]))

        # 创建完整的点序列：原点 + 所有预测点
        full_points = [(0, 0)] + points

        # 提取x和y坐标
        pred_x = [p[0] for p in full_points]
        pred_y = [p[1] for p in full_points]

        # 绘制折线（原点不标记）
        plt.plot(pred_x, pred_y, 'r--', linewidth=2, label='Prediction')

        # 只标记预测点（不包括原点）
        if points:
            pred_x_only = [p[0] for p in points]
            pred_y_only = [p[1] for p in points]
            plt.scatter(pred_x_only, pred_y_only, c='red', s=30, zorder=5)

    # 添加标签和图例
    plt.xlabel('Displacement (mm)', size=12)
    plt.ylabel('Load (kN)', size=12)
    plt.title(f'Curve {i}', size=15)
    plt.grid(alpha=0.3)
    plt.legend()

    # 保存图像
    plt.savefig(os.path.join('curves', f'{i}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"成功生成 {i} 条曲线对比图，已保存至 curves 文件夹！")
