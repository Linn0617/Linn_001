import matplotlib.pyplot as plt
import numpy as np
import os
import math


def color1(value):
    color_bar = (0.5, value, 1)
    return color_bar


def color2(value):
    color_bar = (0, value, 1)
    return color_bar


def color3(value):
    color_bar = (1, value, 0.5)
    return color_bar


def color4(value):
    color_bar = (1, value, 0)
    return color_bar


def gene_pic(data_dir, file_dir1):
    data = np.loadtxt(data_dir, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))

    plt.rcParams['axes.facecolor'] = 'k'
    for i in range(data.shape[0]):
        ln = data[i][0]

        line1 = [[0, 4], [4, 4]]
        line2 = [[0, 4], [7, 7]]
        line3 = [[7, 11], [4, 4]]
        line4 = [[7, 11], [7, 7]]
        line5 = [[4, 4], [0, 4]]
        line6 = [[7, 7], [0, 4]]
        line7 = [[4, 4], [7, 11]]
        line8 = [[7, 7], [7, 11]]

        spring1 = [[0, 0], [4, 7]]
        spring2 = [[11, 11], [4, 7]]
        spring3 = [[4, 7], [0, 0]]
        spring4 = [[4, 7], [11, 11]]

        steel_x1 = [[0, 11], [5, 5]]
        steel_x2 = [[0, 11], [6, 6]]
        steel_y1 = [[5, 5], [0, 11]]
        steel_y2 = [[6, 6], [0, 11]]

        h1 = data[i][3] * 1.16
        h2 = data[i][5] * 1.16
        fc = (data[i][7] - 18) / 24
        s1 = data[i][8] * 140
        s2 = data[i][9] * 140
        s3 = data[i][10] * 140
        s4 = data[i][11] * 140
        k1 = (math.log10(data[i][12]) / 10)
        k2 = (math.log10(data[i][13]) / 10)
        k3 = (math.log10(data[i][14]) / 10)
        k4 = (math.log10(data[i][15]) / 10)
        c1 = ((data[i][16] * 1000) - 20) / 80
        c2 = ((data[i][17] * 1000) - 20) / 80

        if ln > 0:
            plt.figure(figsize=(8, 8))
            plt.fill_between([0, 11], 4, 7, color=color4(fc), alpha=0.5, linewidth=0)
            plt.fill_between([4, 7], 0, 4, color=color4(fc), alpha=0.5, linewidth=0)
            plt.fill_between([4, 7], 7, 11, color=color4(fc), alpha=0.5, linewidth=0)
            plt.plot(line1[0], line1[1], c=color2(h1), linewidth=10)
            plt.plot(line2[0], line2[1], c=color2(c1), linewidth=10)
            plt.plot(line3[0], line3[1], c=color2(h1), linewidth=10)
            plt.plot(line4[0], line4[1], c=color2(c1), linewidth=10)
            plt.plot(line5[0], line5[1], c=color2(c2), linewidth=10)
            plt.plot(line6[0], line6[1], c=color2(h2), linewidth=10)
            plt.plot(line7[0], line7[1], c=color2(c2), linewidth=10)
            plt.plot(line8[0], line8[1], c=color2(h2), linewidth=10)
            plt.plot(steel_x1[0], steel_x1[1], c=color1(s1), linewidth=12)
            plt.plot(steel_x2[0], steel_x2[1], c=color1(s2), linewidth=12)
            plt.plot(steel_y1[0], steel_y1[1], c=color1(s3), linewidth=12)
            plt.plot(steel_y2[0], steel_y2[1], c=color1(s4), linewidth=12)
            plt.plot(spring1[0], spring1[1], c=color3(k1), linewidth=20)
            plt.plot(spring2[0], spring2[1], c=color3(k2), linewidth=20)
            plt.plot(spring3[0], spring3[1], c=color3(k3), linewidth=20)
            plt.plot(spring4[0], spring4[1], c=color3(k4), linewidth=20)
            plt.xticks([])
            plt.yticks([])
            # plt.pause(0.1)
            if os.path.exists(file_dir1) is False:
                os.makedirs(file_dir1)
            os.chdir(file_dir1)
            plt.savefig('%s' % i + '.png', dpi=8, pad_inches=0.0, bbox_inches='tight')
            plt.close()
            orign_dir = r'../'
            os.chdir(orign_dir)
        else:
            plt.figure(figsize=(8, 8))
            plt.xticks([])
            plt.yticks([])
            plt.pause(0.1)
            if os.path.exists(file_dir1) is False:
                os.makedirs(file_dir1)
            os.chdir(file_dir1)
            plt.savefig('%s' % i + '.png', dpi=8, pad_inches=0.0, bbox_inches='tight')
            plt.close()
            orign_dir = r'../'
            os.chdir(orign_dir)


if __name__ == '__main__':
    gene_pic(r'..\train_data_2500\parameters_2500.txt',
             r'.\feature_images')
