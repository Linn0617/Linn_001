import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import RepeatedKFold
import time
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 创建数据集
class MyDataset(Dataset):
    def __init__(self, folder, transform, label_file, parameter_file):
        images = []
        labels = []
        span_x = []
        span_y = []
        width_x = []
        width_y = []

        for i in range(len(os.listdir(folder))):
            images.append(folder + '\\' + str(i) + '.png')
        label = np.loadtxt(label_file)
        for n in range(label.shape[0]):
            labels.append(label[n])
        geometry = np.loadtxt(parameter_file)
        for m in range(geometry.shape[0]):
            span_x.append(geometry[m][1])
            span_y.append(geometry[m][2])
            width_x.append(geometry[m][4])
            width_y.append(geometry[m][6])

        self.folder = folder
        self.transforms = transform
        self.images = images
        self.labels = labels
        self.span_x = span_x
        self.span_y = span_y
        self.width_x = width_x
        self.width_y = width_y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]
        img_label = self.labels[item]
        img = Image.open(img_file).convert('RGB')
        img = self.transforms(img)
        data1 = torch.full([3, 49, 1], self.span_y[item])
        data2 = torch.full([3, 49, 1], self.width_y[item])
        img_1 = torch.cat([img, data1, data2], dim=2)
        data3 = torch.full([3, 1, 51], self.span_x[item])
        data4 = torch.full([3, 1, 51], self.width_x[item])
        img_mix = torch.cat([img_1, data3, data4], dim=1)
        return img_mix, img_label


class Dataset(Dataset):
    def __init__(self, folder, index):
        images = []
        labels = []
        for j in range(len(index)):
            src, trg = folder[index[j]]
            images.append(src)
            labels.append(trg)
        self.folder = folder
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        img_label = self.labels[item]
        return img, img_label


loss_func = torch.nn.MSELoss(reduction='mean')  # 损失函数


# 训练模型
def train_epoch(model_S, training_data, optimizer, decay_optim, epoch, filename1, filename2):
    ''' Epoch operation in training phase'''
    global global_step
    # global step
    model_S.train()
    total_loss = 0
    loss_pred_all, loss_sta_all = 0, 0
    step = 0
    for src_seq, trg_seq in training_data:
        src_seq = src_seq.cuda().float()
        trg_seq = trg_seq.cuda().float()
        src_seq = src_seq.squeeze()
        optimizer.zero_grad()
        pred_labeled = model_S(src_seq)

        pred_loss = loss_func(pred_labeled, trg_seq)

        pred_loss.backward()
        # optimizer.step()
        optimizer.step()

        loss_pred_all += pred_loss.item() * src_seq.size(0)
        step += src_seq.size(0)
        a = pred_labeled.cpu().detach().numpy()
        b = trg_seq.cpu().detach().numpy()
        with open(filename1, 'a') as f1:
            np.savetxt(f1, a, fmt='%f', delimiter=',')
        f1.close()
        with open(filename2, 'a') as f2:
            np.savetxt(f2, b, fmt='%f', delimiter=',')
        f2.close()
    return loss_pred_all / step


# 验证模型
def eval_epoch(model, validation_data, filename1, filename2):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    with torch.no_grad():
        # for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
        total_loss = 0.0
        step_valid = 0
        for src_seq_v, trg_seq_v in validation_data:
            src_seq_v = src_seq_v.cuda().float()
            trg_seq_v = trg_seq_v.cuda().float()
            # src_seq_v = src_seq_v.squeeze()
            # print(src_seq_v, trg_seq_v)
            pred_seq_v = model(src_seq_v)
            loss = loss_func(pred_seq_v, trg_seq_v)
            print('each loss', loss)
            total_loss += loss.item() * src_seq_v.size(0)
            step_valid += src_seq_v.size(0)
            print('each size', step_valid)
            c = pred_seq_v.cpu().numpy()
            d = trg_seq_v.cpu().numpy()
            with open(filename1, 'a') as f3:
                # 四个参数依次为文件名、数组、数据类型（浮点型）、分隔符（逗号）
                np.savetxt(f3, c, fmt='%f', delimiter=',')
            f3.close()
            with open(filename2, 'a') as f4:
                # 四个参数依次为文件名、数组、数据类型（浮点型）、分隔符（逗号）
                np.savetxt(f4, d, fmt='%f', delimiter=',')
            f4.close()
        print('total loss', total_loss)
        # print('size', step_valid)
    return total_loss / step_valid


def train(model, all_data, optimizer, decay_optim, device, epoch):
    log_file = 'train.log'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch,n, train_loss, valid_loss\n')

    def print_performances(header, loss, start_time):
        print('  - {header:15} : {loss: 8.10f}, ' \
              'elapse: {elapse:3.3f} min'.format(header=f"({header})", loss=loss,
                                                 elapse=(time.time() - start_time) / 60))

    train_losses = []
    valid_losses = []

    for epoch_i in range(epoch):
        print('[epoch %s' % epoch_i)
        x = all_data
        ss = RepeatedKFold(n_splits=5, n_repeats=1)
        n = 0
        for train_index, test_index in ss.split(x):
            filename1 = f"{0}_{epoch_i}_{n}.txt"
            dir1 = r'./' + '\\' + filename1  # 响应真实值保存路径
            filename2 = f"{1}_{epoch_i}_{n}.txt"
            dir2 = r'./' + '\\' + filename2  # 响应预测值保存路径
            train_data = Dataset(folder=all_data, index=train_index)
            valid_data = Dataset(folder=all_data, index=test_index)
            train_loader = DataLoader(dataset=train_data, batch_size=80, shuffle=True)
            valid_loader = DataLoader(dataset=valid_data, batch_size=20)
            print('training sample:%s' % train_loader.__len__())
            print('validation sample:%s' % valid_loader.__len__())
            start = time.time()
            train_loss_pred = train_epoch(model, train_loader, optimizer, decay_optim, epoch_i, filename1, filename2)
            print_performances('Training Loss Pred', train_loss_pred, start)
            train_losses += [train_loss_pred]
            decay_optim.step()
            start = time.time()
            valid_loss = eval_epoch(model, valid_loader, filename1, filename2)
            print_performances('Validation Loss Pred', valid_loss, start)
            valid_losses += [valid_loss]
            save_dir1 = r'.\predict'
            save_dir2 = r'.\true'
            if os.path.exists(save_dir1) is False:
                os.makedirs(save_dir1)
            if os.path.exists(save_dir2) is False:
                os.makedirs(save_dir2)
            shutil.move(dir1, save_dir1)
            shutil.move(dir2, save_dir2)
            checkpoint = {'epoch': epoch_i, 'model': model.state_dict()}
            model_name1 = 'best_train.train'
            model_name2 = 'best_valid.train'
            if train_loss_pred <= min(train_losses):
                torch.save(checkpoint, model_name1)
                print('    - [Info] The train checkpoint file has been updated.')
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, model_name2)
                print('    - [Info] The valid checkpoint file has been updated.')
            with open(log_file, 'a') as log_f:
                log_f.write('{epoch},{n},{train_loss: 8.10f},{valid_loss: 8.10f},\n'.format(
                    epoch=epoch_i, n=n, train_loss=train_loss_pred, valid_loss=valid_loss))
            n = n + 1


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    all_folder = r'.\feature_images\feature_images'  # 配筋特征图的路径
    all_txt = 'train_data_2500/labels_2500.txt'
    parameters = 'train_data_2500/parameters_2500.txt'
    all_data = MyDataset(folder=all_folder, label_file=all_txt, parameter_file=parameters, transform=transform)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 1e-3
    epochs = 2000
    net = models.resnet18(num_classes=8)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-09)
    decay_optim = optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

    train(net, all_data, optimizer, decay_optim, device, epochs)


if __name__ == '__main__':
    from datetime import datetime

    start = datetime.now()
    main()
    end = datetime.now()
    f = open('train_cost.txt', 'a+')  # 记录模型训练用时
    f.write(start.strftime("%m.%d.%H.%M\n"))
    f.write(end.strftime("%m.%d.%H.%M\n"))
    f.close()
