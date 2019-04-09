import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import h5py

class VideoDataset(Dataset):
    def __init__(self, config, dataset='aicity', phase='train', preprocess=False):
        self.root_dir, self.output_dir = config.db_dir(dataset)

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 112
        self.resize_width = 112
        self.config = config
        self.phase = phase
        self.prepare_data()

        if preprocess:
            self.create_dataset()

        if phase == 'test':
            self.dataset = h5py.File(config.prepare_hdf5_dir + '/' + config.dataname)

        # processed_vid = self.dataset['train/anomaly/0'][()]
        # fig = plt.figure()
        # ims = []
        # for ii in range(0, np.shape(processed_vid)[1]):
        #     im = plt.imshow(np.dstack((processed_vid[0][ii], processed_vid[1][ii], processed_vid[2][ii])), animated=True)
        #     ims.append([im])
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
        # plt.show()

    def box_intersect(self, box1, box2):
        xmin = max(box1[0], box2[0])
        xmax = min(box1[2], box2[2])
        ymin = max(box1[1], box2[1])
        ymax = min(box1[3], box2[3])
        if (xmax - xmin < 0) or (ymax - ymin < 0):
            return 0
        return (xmax - xmin) * (ymax - ymin)

    def interval_intersect(self, l1, r1, l2, r2):
        lmax = max(l1, l2)
        rmin = min(r1, r2)
        if (lmax > rmin):
            return 0
        return rmin - lmax
    
    def prepare_normal(self, dataset, anomaly_videos, normal_videos, raw_set):
        
        for i in range(0, self.config.test_normal_size):
            video_id = i % 100 + 1
            filename = self.root_dir + '/' + str(video_id) + '.mp4'
            video_w, video_h = self.config.video_size

            capture = cv2.VideoCapture(filename)
            video_max_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
            #select random interval
            l = random.randint(0, video_max_len - self.config.prepare_len)
            r = l + self.config.prepare_len
            x1 = random.randint(0, video_w - self.config.prepare_crop_size[0])
            y1 = random.randint(0, video_h - self.config.prepare_crop_size[1])
            x2 = x1 + self.config.prepare_crop_size[0]
            y2 = y1 + self.config.prepare_crop_size[1]

            while self.checkLabel(video_id, l, r, x1, y1, x2, y2):
                l = random.randint(0, video_max_len - self.config.prepare_len)
                r = l + self.config.prepare_len
                x1 = random.randint(0, video_w - self.config.prepare_crop_size[0])
                y1 = random.randint(0, video_h - self.config.prepare_crop_size[1])
                x2 = x1 + self.config.prepare_crop_size[0]
                y2 = y1 + self.config.prepare_crop_size[1]

            capture.set(cv2.CAP_PROP_POS_FRAMES, l - 1)
            processed_vid = []
            for frame_id in range(l, r, self.config.prepare_len // self.config.prepare_len_sample):
                group_id = 0
                # while capture_frame < frame_id:
                #     ret, frame = capture.read()
                #     capture_frame += 1
                ret, frame = capture.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_vid.append(rgb[y1: y2, x1: x2, :])

            capture.release()    
            data_hdf5 = dataset.create_dataset(name=str(i), data=processed_vid, shape=np.shape(processed_vid), dtype=int)
            print(data_hdf5.name)
            data_hdf5.attrs['video'] = video_id
            data_hdf5.attrs['anomaly'] = 0

    def prepare_anomaly(self, dataset, anomaly_videos, normal_videos, raw_set):
        
        for i in range(0, self.config.test_anomaly_size):
            video_id = anomaly_videos[i % len(anomaly_videos)]
            filename = self.root_dir + '/' + str(video_id) + '.mp4'

            capture = cv2.VideoCapture(filename)
            video_max_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            #select random interval
            l = random.randint(raw_set[video_id][0] * self.config.video_fps, raw_set[video_id][1] * self.config.video_fps - self.config.prepare_len)
            r = l + self.config.prepare_len

            anomaly_box = raw_set[video_id][2:6]
            low = min(anomaly_box[0], max(anomaly_box[2] - self.config.crop_size[0], 0))
            high = anomaly_box[0]
            x1 = np.random.randint(low, high, 1)[0]
            low = min(anomaly_box[1], max(anomaly_box[3] - self.config.crop_size[1], 0))
            high = anomaly_box[1]
            y1 = np.random.randint(low, high, 1)[0]
            
            x2, y2 = np.array([x1, y1]) + np.array(self.config.crop_size)
            x2 = min(x2, self.config.video_size[0])
            y2 = min(y2, self.config.video_size[1])
            print('%d %d %d %d %d %d' % (l, r, x1, y1, x2, y2))

            capture.set(cv2.CAP_PROP_POS_FRAMES, l - 1)
            processed_vid = []
            for frame_id in range(l, r, self.config.prepare_len // self.config.prepare_len_sample):
                ret, frame = capture.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_vid.append(rgb[y1: y2, x1: x2, :])

            capture.release()    
            data_hdf5 = dataset.create_dataset(name=str(i), data=processed_vid, shape=np.shape(processed_vid), dtype=int)
            print(data_hdf5.name)
            data_hdf5.attrs['video'] = video_id
            data_hdf5.attrs['anomaly'] = 1

            #debug
            # print(np.shape(processed_vid))
            # fig = plt.figure()
            # ims = []
            # for ii in range(0, np.shape(processed_vid)[1]):
            #     im = plt.imshow(np.dstack((processed_vid[0][ii], processed_vid[1][ii], processed_vid[2][ii])), animated=True)
            #     ims.append([im])
            # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
            # plt.show()
            
            # print('finish')

    def prepare_data(self):

        # l r x1 y1 x2 y2
        raw_txt = self.root_dir + '/train-anomaly.txt'
        self.raw_set = {}
        with open(raw_txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_id, l, r, x1, y1, x2, y2 = [int(value) for value in line.split(' ')]
                if video_id in self.raw_set.keys():
                    self.raw_set[video_id].append([l * self.config.video_fps, r * self.config.video_fps, x1, y1, x2, y2])
                else:
                    self.raw_set[video_id] = [[l * self.config.video_fps, r * self.config.video_fps, x1, y1, x2, y2]]
        
        self.anomaly_videos = list(self.raw_set.keys())
        self.normal_videos = list(set(range(1, 101)) - set(self.anomaly_videos))

    def create_dataset(self):
        hdf5_dataset = h5py.File(self.config.prepare_hdf5_dir + '/' + self.config.dataname, 'w')
        train_set = hdf5_dataset.create_group('test')
        train_set.attrs['normal_length'] = self.config.normal_size
        train_set.attrs['anomaly_length'] = self.config.anomaly_size
        normal_set = train_set.create_group('normal')
        anomaly_set = train_set.create_group('anomaly')

        self.prepare_normal(normal_set, self.anomaly_videos, self.normal_videos, self.raw_set)
        self.prepare_anomaly(anomaly_set, self.anomaly_videos, self.normal_videos, self.raw_set)

        hdf5_dataset.close()

    def loadRandomCroppedVehicle(self, augmented_path):
        paths = os.listdir(augmented_path)
        index = random.randint(0, len(paths) - 1)
        #print(paths[index])
        vehicle = cv2.cvtColor(cv2.imread(augmented_path + '/' + paths[index]), cv2.COLOR_BGR2RGB)
        return vehicle

    def mergeAugmentedVehicle(self, augmented_vehicle, buffer, posx, posy):
        lx = max(posx, 0)
        rx = min(posx + augmented_vehicle.shape[1], buffer.shape[1])
        ly = max(posy, 0)
        ry = min(posy + augmented_vehicle.shape[0], buffer.shape[0])
        for j in range(lx, rx):
            for i in range(ly, ry):
                if augmented_vehicle[i - posy][j - posx] > 0:
                    buffer[i][j] = augmented_vehicle[i - posy][j - posx]
        return buffer

    def checkLabel(self, video_id, l, r, x1, y1, x2, y2):
        #print(l, ' ', r, ' ', x1, ' ', y1, ' ', x2, ' ', y2)
        if video_id in self.normal_videos:
            label = 0
        else:
            #anomaly interval and bounding box of anomaly vehicle
            for i in range(0, len(self.raw_set[video_id])):
                gt_l, gt_r, gt_x1, gt_y1, gt_x2, gt_y2 = self.raw_set[video_id][i]
                ratio = self.interval_intersect(l, r, gt_l, gt_r) / (r - l)
                if ratio < 0.5:
                    label = 0
                else:
                    ratio = self.box_intersect((x1, y1, x2, y2), (gt_x1, gt_y1, gt_x2, gt_y2)) / ((gt_x2 - gt_x1) * (gt_y2 - gt_y1))
                    if ratio < 0.6:
                        label = 0
                    else:
                        label = 1
                if label == 1:
                    break
        
        #print(label)
        return label

    def __len__(self):
        return self.config.step_per_epoch

    def __getitem__(self, index):
        if self.phase == 'train':
            video_id = index + 1
            # video_id = 93
            filename = self.root_dir + '/' + str(video_id) + '.mp4'
            #print(filename)
            capture = cv2.VideoCapture(filename)
            video_max_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            #select random interval
            l = random.randint(0, video_max_len - self.config.prepare_len)
            r = l + self.config.prepare_len

            #split region in an interval
            video_w, video_h = self.config.video_size
            batch = []
            labels = []
            cropped_regions = []
            num_w_window = (video_w - self.config.prepare_crop_size[0]) // self.config.step_w + 1
            num_h_window = (video_h - self.config.prepare_crop_size[1]) // self.config.step_h + 1
            #print(num_w_window, ' ', num_h_window)
            for i in range(0, num_h_window * num_w_window):
                batch.append([])
                cropped_regions.append([])

            capture_frame = -1
            capture.set(cv2.CAP_PROP_POS_FRAMES, l - 1)
            for frame_id in range(l, r, self.config.prepare_len // self.config.prepare_len_sample):
                group_id = 0
                # while capture_frame < frame_id:
                #     ret, frame = capture.read()
                #     capture_frame += 1
                ret, frame = capture.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for x in range(0, video_w - self.config.prepare_crop_size[0] + 1, self.config.step_w):
                    for y in range(0, video_h - self.config.prepare_crop_size[1] + 1, self.config.step_h):
                        if (cropped_regions[group_id] == []):
                            x1 = random.randint(x - 40, x + 40)
                            y1 = random.randint(y - 40, y + 40)
                            x2 = x1 + self.config.prepare_crop_size[0]
                            y2 = y1 + self.config.prepare_crop_size[1]
                            x1 = max(x1, 0)
                            y1 = max(y1, 0)
                            x2 = min(x2, self.config.video_size[0])
                            y2 = min(y2, self.config.video_size[1])
                            cropped_regions[group_id] = [x1, y1, x2, y2]
                        else:
                            x1, y1, x2, y2 = cropped_regions[group_id]
                        sample = rgb[y1: y2, x1: x2, :]
                        batch[group_id].append(cv2.resize(sample, self.config.resized_shape))
                        group_id += 1
            
            for cropped_region in cropped_regions:
                x1, y1, x2, y2 = cropped_region
                labels.append(self.checkLabel(video_id, l, r, x1, y1, x2, y2))
                #labels.append(1)
        
            capture.release()

            batch = np.array(batch, dtype=np.float32)
            #num_region x len_interval x cropped_h x cropped_w x channels -> num_region x channels x len_interval x cropped_h x cropped_w
            batch = np.swapaxes(batch, 3, 4)
            batch = np.swapaxes(batch, 2, 3)
            batch = np.swapaxes(batch, 1, 2)
            labels = np.array(labels, dtype=np.int64)

            # data augmentation (optional)
            # for k in range(0, batch.shape[0]):
            #     turnToAnomaly = random.randint(0, 2)
            #     if turnToAnomaly == 0:
            #         buffer = batch[k]
            #         augmented_vehicle = self.loadRandomCroppedVehicle(self.config.prepare_cropped_vehicles)

            #         #random resize
            #         scale = random.uniform(0.5, 1.5)
            #         h, w, _ = augmented_vehicle.shape
            #         augmented_vehicle = cv2.resize(augmented_vehicle, (int(w * scale), int(h * scale)))
            #         #random place
            #         # posx = random.randint(-augmented_vehicle.shape[1]//2, self.config.prepare_crop_size[0] - augmented_vehicle.shape[1] // 2)
            #         # posy = random.randint(-augmented_vehicle.shape[0]//2, self.config.prepare_crop_size[1] - augmented_vehicle.shape[0] // 2)
            #         posx = random.randint(0, self.config.resized_shape[0] - augmented_vehicle.shape[1])
            #         posy = random.randint(0, self.config.resized_shape[1] - augmented_vehicle.shape[0])
            #         # print(posx, ' ', posy)
            #         # print('vehicle size: ', augmented_vehicle.shape)
            #         for i in range(0, 3):
            #             for j in range(0, self.config.prepare_len_sample):
            #                 buffer[i][j] = self.mergeAugmentedVehicle(augmented_vehicle[:, :, i], buffer[i][j], posx, posy)

            #         # processed_vid = batch[k].astype(int)
            #         # fig = plt.figure()
            #         # ims = []
            #         # for ii in range(0, np.shape(processed_vid)[1]):
            #         #     im = plt.imshow(np.dstack((processed_vid[0][ii], processed_vid[1][ii], processed_vid[2][ii])), animated=True)
            #         #     ims.append([im])
            #         # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
            #         # plt.show()
                    
            #         box1 = [0, 0, self.config.prepare_crop_size[0], self.config.prepare_crop_size[1]]
            #         box2 = [posx, posy, posx + augmented_vehicle.shape[1], posy + augmented_vehicle.shape[0]]
            #         #soft assign label
            #         #label = 1.0 * self.box_intersect(box1, box2) / (1.0 * augmented_vehicle.shape[0] * augmented_vehicle.shape[1])
            #         labels[k] = 1

            # # video_segment = batch
            # # for i in range(0, len(video_segment)):
            # #     processed_vid = video_segment[i].astype(int)
            # #     print(labels[i])
            # #     fig = plt.figure()
            # #     ims = []
            # #     for ii in range(0, np.shape(processed_vid)[1]):
            # #         im = plt.imshow(np.dstack((processed_vid[0][ii], processed_vid[1][ii], processed_vid[2][ii])), animated=True)
            # #         ims.append([im])
            # #     ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
            # #     plt.show()
            # # print(batch.shape, ' ', labels.shape)
        else:
            if index < self.dataset[self.phase].attrs['normal_length']:
                set_name = 'normal'
                set_id = index
            else:
                set_name = 'anomaly'
                set_id = index - self.dataset[self.phase].attrs['normal_length']
            buffer = np.array(self.dataset[self.phase + '/' + set_name + '/' + str(set_id)][()], dtype=np.dtype('float32'))
            label = self.dataset[self.phase + '/' + set_name + '/' + str(set_id)].attrs['anomaly']

            for i in range(0, self.config.prepare_len_sample):
                buffer[i] = cv2.resize(buffer[i], self.config.resized_shape)

            batch = np.array(buffer, dtype=np.float32)
            batch = np.swapaxes(batch, 2, 3)
            batch = np.swapaxes(batch, 1, 2)
            batch = np.swapaxes(batch, 0, 1)
            labels = np.array([label], dtype=np.int64)
        return batch, labels

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from torch.utils.data import DataLoader
    import config_net
    train_data = VideoDataset(config_net, dataset='aicity', preprocess=False)    
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)

    for inputs, labels in train_dataloader:
        print(inputs.shape, ' ', labels.shape)
        #break

# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     import config_net
#     train_data = VideoDataset(config_net, dataset='aicity', split='train', clip_len=15, preprocess=False)
#     train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

#     for i, sample in enumerate(train_loader):
#         inputs = sample[0]
#         labels = sample[1]
#         print(inputs.size())
#         print(labels)

#         if i == 1:
#             break