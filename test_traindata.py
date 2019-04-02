from network import R2Plus1D_model
import torch
from torch import nn
import config_net as config
import imageio
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import h5py

class TestDataset(Dataset):

    def __init__(self):
        self.dataset = h5py.File(config.prepare_hdf5_dir + '/AiCity2.hdf5', 'r')

    def __len__(self):
        return self.dataset['train'].attrs['normal_length'] + self.dataset['train'].attrs['anomaly_length']

    def __getitem__(self, index):
        set_name = 'anomaly'
        if index < self.dataset['train'].attrs['normal_length']:
            set_name = 'normal'
            set_id = index
        else:
            set_name = 'anomaly'
            set_id = index - self.dataset['train'].attrs['normal_length']

        buffer = np.array(self.dataset['train/' + set_name + '/' + str(set_id)][()], dtype=np.dtype('float32'))
        label = self.dataset['train/' + set_name + '/' + str(set_id)].attrs['anomaly']
        
        temp = [[], [], []]
        for i in range(0, 3):
            for j in range(0, config.prepare_len_sample):
                temp[i].append(cv2.resize(buffer[i][j], (config.crop_size, config.crop_size)))   
        buffer = np.array(temp)

        # processed_vid = buffer.astype(int)
        # fig = plt.figure()
        # ims = []
        # for ii in range(0, np.shape(processed_vid)[1]):
        #     im = plt.imshow(np.dstack((processed_vid[0][ii], processed_vid[1][ii], processed_vid[2][ii])), animated=True)
        #     ims.append([im])
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
        # plt.show()

        # one_hot_label = np.zeros([2], dtype=int)
        # one_hot_label[label] = 1
        return torch.from_numpy(buffer), torch.from_numpy(np.array(label))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = R2Plus1D_model.R2Plus1DClassifier(2, (2, 2, 2, 2), pretrained=True).to(device)
model.eval()

data = 'data/AiCity2.hdf5'
test_dataloader = DataLoader(TestDataset(), batch_size=1, shuffle=False, num_workers=1)

running_corrects = 0
index = 0
normal_acc = 0
anomaly_acc = 0
for inputs, labels in test_dataloader:
    index += 1
    print(index)
    print(np.shape(inputs))
    # move inputs and labels to the device the training is taking place on
    # print(np.shape(processed_vid))
    inputs = Variable(inputs, requires_grad=False).to(device)
    labels = Variable(labels).to(device)

    outputs = model(inputs)

    probs = nn.Softmax(dim=1)(outputs)
    preds = torch.max(probs, 1)[1]
    print(preds, labels.data)
    running_corrects += torch.sum(preds == labels.data)
    if (labels.data.cpu().numpy()[0] == 0 and torch.sum(preds == labels.data)):
        normal_acc += 1
    if (labels.data.cpu().numpy()[0] == 1 and torch.sum(preds == labels.data)):
        anomaly_acc += 1

epoch_acc = running_corrects.double() / len(test_dataloader.dataset)
print(epoch_acc)
print(normal_acc)
print(anomaly_acc)
