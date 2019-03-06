import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import config_net

class VideoDataset(Dataset):
    def __init__(self, dataset='aicity', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = config_net.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='aicity', split='train', clip_len=15, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break