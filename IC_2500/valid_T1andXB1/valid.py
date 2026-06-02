import math
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import time
from shutil import copy
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 绘制配筋特征图
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


def gene_pic(A, B, save_dir1, h1, h2):
    plt.rcParams['axes.facecolor'] = 'k'
    ln = 1

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

    if ln > 0:
        plt.figure(figsize=(8, 8))
        plt.fill_between([0, 11], 4, 7, color=color4((fc - 18) / 24), alpha=0.5, linewidth=0)
        plt.fill_between([4, 7], 0, 4, color=color4((fc - 18) / 24), alpha=0.5, linewidth=0)
        plt.fill_between([4, 7], 7, 11, color=color4((fc - 18) / 24), alpha=0.5, linewidth=0)
        plt.plot(line1[0], line1[1], c=color2(h1 * 1.16), linewidth=10)
        plt.plot(line2[0], line2[1], c=color2(((c1 * 1000) - 20) / 80), linewidth=10)
        plt.plot(line3[0], line3[1], c=color2(h1 * 1.16), linewidth=10)
        plt.plot(line4[0], line4[1], c=color2(((c1 * 1000) - 20) / 80), linewidth=10)
        plt.plot(line5[0], line5[1], c=color2(((c2 * 1000) - 20) / 80), linewidth=10)
        plt.plot(line6[0], line6[1], c=color2(h2 * 1.16), linewidth=10)
        plt.plot(line7[0], line7[1], c=color2(((c2 * 1000) - 20) / 80), linewidth=10)
        plt.plot(line8[0], line8[1], c=color2(h2 * 1.16), linewidth=10)
        plt.plot(steel_x1[0], steel_x1[1], c=color1(A[0]), linewidth=12)
        plt.plot(steel_x2[0], steel_x2[1], c=color1(A[1]), linewidth=12)
        plt.plot(steel_y1[0], steel_y1[1], c=color1(A[2]), linewidth=12)
        plt.plot(steel_y2[0], steel_y2[1], c=color1(A[3]), linewidth=12)
        plt.plot(spring1[0], spring1[1], c=color3(B[0]), linewidth=20)
        plt.plot(spring2[0], spring2[1], c=color3(B[1]), linewidth=20)
        plt.plot(spring3[0], spring3[1], c=color3(B[2]), linewidth=20)
        plt.plot(spring4[0], spring4[1], c=color3(B[3]), linewidth=20)
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.1)
        if os.path.exists(save_dir1) is False:
            os.makedirs(save_dir1)
        os.chdir(save_dir1)
        plt.savefig('%s' % 0 + '.png', dpi=8, pad_inches=0.0, bbox_inches='tight')
        plt.close()
        orign_dir = r'../'
        os.chdir(orign_dir)
    else:
        plt.figure(figsize=(8, 8))
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.1)
        if os.path.exists(save_dir1) is False:
            os.makedirs(save_dir1)
        os.chdir(save_dir1)
        plt.savefig('%s' % 0 + '.png', dpi=8, pad_inches=0.0, bbox_inches='tight')
        plt.close()
        orign_dir = r'../'
        os.chdir(orign_dir)


# 构建数据集
class MyDataset(Dataset):
    def __init__(self, folder, transform, l1, b1, l2, b2):
        images = []
        spans1 = []
        spans2 = []
        widths1 = []
        widths2 = []
        for i in range(len(os.listdir(folder))):
            images.append(folder + '\\' + str(i) + '.png')
        self.folder = folder
        self.transforms = transform
        self.images = images
        self.spans1 = l1
        self.spans2 = l2
        self.widths1 = b1
        self.widths2 = b2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]
        img = Image.open(img_file).convert('RGB')
        img = self.transforms(img)
        data1 = torch.full([3, 49, 1], self.spans2)
        data2 = torch.full([3, 49, 1], self.widths2)
        img_1 = torch.cat([img, data1, data2], dim=2)
        data3 = torch.full([3, 1, 51], self.spans1)
        data4 = torch.full([3, 1, 51], self.widths1)
        img_mix = torch.cat([img_1, data3, data4], dim=1)
        return img_mix


# 用训练的预测模型预测结构响应
def eval_epoch(model, validation_data):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    with torch.no_grad():
        # for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
        for src_seq_v in validation_data:
            src_seq_v = src_seq_v.cuda().float()
            pred_seq_v = model(src_seq_v)
    return pred_seq_v


def write_result(folder_series, l1, l2, b1, h1, b2, h2, tA11, tA12, tA13, tA14,
                 ypred1, ypred2, ypred3, ypred4, ypred5, ypred6, ypred7, ypred8):
    with open(folder_series, 'a') as f:
        f.write(
            '{l1}, {l2}, {b1}, {h1}, {b2}, {h2}, {tA11}, {tA12}, {tA13}, {tA14}, {ypred1}, {ypred2}, {ypred3},'
            '{ypred4}, {ypred5}, {ypred6}, {ypred7}, {ypred8}'.format(
                l1=l1, l2=l2, b1=b1, h1=h1, b2=b2, h2=h2, tA11=tA11, tA12=tA12, tA13=tA13, tA14=tA14,
                ypred1=ypred1, ypred2=ypred2, ypred3=ypred3, ypred4=ypred4, ypred5=ypred5, ypred6=ypred6, ypred7=ypred7, ypred8=ypred8))
        f.write('\n')


def write_result2(folder_series, ypred1, ypred2, ypred3, ypred4, ypred5, ypred6, ypred7, ypred8):
    with open(folder_series, 'a') as f:
        f.write('{ypred1}, {ypred2}, {ypred3},{ypred4}, {ypred5}, {ypred6}, {ypred7}, {ypred8}'.format(
            ypred1=ypred1, ypred2=ypred2, ypred3=ypred3, ypred4=ypred4, ypred5=ypred5, ypred6=ypred6, ypred7=ypred7, ypred8=ypred8))
        f.write('\n')


def main(A, B, save_dir1, l1, l2, b1, h1, b2, h2):
    gene_pic(A, B, save_dir1, h1, h2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valid_data = MyDataset(folder=save_dir1, transform=transform, l1=l1, b1=b1, l2=l2, b2=b2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1)
    net = models.resnet18(num_classes=8)
    net = net.to(device)
    checkpoint = torch.load(r'../best_train.train')  # 预测模型路径
    net.load_state_dict(checkpoint['model'])
    pred_y = eval_epoch(net, valid_loader)
    ypred = pred_y.detach().cpu().squeeze(0).numpy()
    print(ypred)
    return ypred


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # work_time = time.strftime("%m_%d", time.localtime())
    # pid = str(os.getpid())
    i = 0
    txt = np.loadtxt('parameter_full_size_T1andXB1.txt')
    for i in range(2):
        save_dir1 = r'./1c'  # 保存的路径
        l1 = txt[i][1]
        l2 = txt[i][2]
        h1 = txt[i][3]
        b1 = txt[i][4]
        h2 = txt[i][5]
        b2 = txt[i][6]
        fc = txt[i][7]
        tA11 = txt[i][8]
        tA12 = txt[i][9]
        tA13 = txt[i][10]
        tA14 = txt[i][11]
        k1 = txt[i][12]
        k2 = txt[i][13]
        k3 = txt[i][14]
        k4 = txt[i][15]
        c1 = txt[i][16]
        c2 = txt[i][17]
        print(i)

        A = [tA11 * 140, tA12 * 140, tA13 * 140, tA14 * 140]
        B = [math.log10(k1) / 10, math.log10(k2) / 10, math.log10(k3) / 10, math.log10(k4) / 10]
        ypred1, ypred2, ypred3, ypred4, ypred5, ypred6, ypred7, ypred8 = main(A, B, save_dir1, l1, l2, b1, h1, b2, h2)
        write_result(r'./pred_result_valid.txt', l1, l2, h1, b1, h2, b2, tA11, tA12, tA13, tA14,
                     ypred1, ypred2, ypred3, ypred4, ypred5, ypred6, ypred7, ypred8)
        write_result2(r'./pred_result_valid_2.txt', ypred1, ypred2, ypred3, ypred4, ypred5, ypred6, ypred7, ypred8)

        i = i + 1
    time.sleep(1)
