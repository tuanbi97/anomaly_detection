import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import h5py

class VideoDataset(Dataset):
    def __init__(self, config, dataset='aicity', split='train', preprocess=False):
        self.root_dir, self.output_dir = config.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if preprocess == True:
            self.prepare_data(config)

    def box_intersect(self, box1, box2):
        xmin = max(box1[0], box2[0])
        xmax = min(box1[2], box2[2])
        ymin = max(box1[1], box2[1])
        ymax = min(box1[3], box2[3])
        if (xmax - xmin < 0) or (ymax - ymin < 0):
            return 0
        return (xmax - xmin) * (ymax - ymin)
    
    def prepare_normal(self, config, dataset, anomaly_videos, normal_videos, raw_set):
        crop_size = config.prepare_crop_size
        #prepare_train
        #get 1000 normal samples from abnormal videos
        for i in range(0, 1000):
            video_id = anomaly_videos[np.random.randint(0, len(anomaly_videos), 1)[0]]

            l, r = raw_set[video_id][0], raw_set[video_id][1]
            anomaly_box = raw_set[video_id][2:6]
            print(anomaly_box)

            #find random normal interval
            for j in range(0, 100):
                is_anomaly = False
                lt = np.random.randint(0, config.prepare_video_max_len, 1)[0]
                if lt >= l:
                    lt +=  r - l
                rt = lt + config.prepare_len
                if rt > config.prepare_video_max_len:
                    is_anomaly = True
                if lt < l and rt > l and 1.0 * (rt - l) / config.prepare_len >= 0.5:
                    is_anomaly = True
                if rt > r and lt < r and 1.0 * (r - lt) / config.prepare_len >= 0.5:
                    is_anomaly = True
                if l <= lt and rt <= r:
                    is_anomaly = True
                if is_anomaly == False:
                    break
            
            x1 = np.random.randint(0, config.video_size[0] - crop_size[0], 1)[0]
            y1 = np.random.randint(0, config.video_size[1] - crop_size[1], 1)[0]
            x2, y2 = np.array([x1, y1]) + np.array(crop_size)
            while is_anomaly == True:
                intersect = self.box_intersect([x1, y1, x2, y2], anomaly_box)
                if intersect / ((x2 - x1) *(y2 - y1)) < 0.5:
                    is_anomaly = False
                x1 = np.random.randint(0, config.video_size[0] - crop_size[0], 1)[0]
                y1 = np.random.randint(0, config.video_size[1] - crop_size[1], 1)[0]
                x2, y2 = np.array([x1, y1]) + np.array(crop_size)
            print('%d %d %d %d %d %d' % (lt, rt, x1, y1, x2, y2))

            #process video
            filename = self.root_dir + '/' + str(video_id) + '.mp4'
            vid = imageio.get_reader(filename, 'ffmpeg')
            processed_vid = [[], [], []]
            for frame_id in range(lt, rt, int(config.prepare_len / config.prepare_len_sample)):
                rgb = vid.get_data(frame_id)
                temp_rgb = rgb[y1:y2, x1:x2, :]
                temp_rgb = np.swapaxes(temp_rgb, 1, 2)
                temp_rgb = np.swapaxes(temp_rgb, 0, 1)
                #print(np.shape(temp_rgb))
                for j in range(0, 3):
                    processed_vid[j].append(temp_rgb[j])
            data_hdf5 = dataset.create_dataset(name=str(i), data=processed_vid, shape=np.shape(processed_vid), dtype=int)
            print(data_hdf5.name)
            data_hdf5.attrs['video'] = video_id
            data_hdf5.attrs['anomaly'] = 0

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

        #get 2000 normal samples from normal videos
        for i in range(1000, 3000):
            video_id = normal_videos[np.random.randint(0, len(normal_videos), 1)[0]]
            lt = np.random.randint(0, config.prepare_video_max_len - config.prepare_len, 1)[0]
            rt = lt + config.prepare_len
            x1 = np.random.randint(0, config.video_size[0] - crop_size[0], 1)[0]
            y1 = np.random.randint(0, config.video_size[1] - crop_size[1], 1)[0]
            x2, y2 = np.array([x1, y1]) + np.array(crop_size)
            print('%d %d %d %d %d %d' % (lt, rt, x1, y1, x2, y2))

            #process video
            filename = self.root_dir + '/' + str(video_id) + '.mp4'
            vid = imageio.get_reader(filename, 'ffmpeg')
            processed_vid = [[], [], []]
            for frame_id in range(lt, rt, int(config.prepare_len / config.prepare_len_sample)):
                rgb = vid.get_data(frame_id)
                temp_rgb = rgb[y1:y2, x1:x2, :]
                temp_rgb = np.swapaxes(temp_rgb, 1, 2)
                temp_rgb = np.swapaxes(temp_rgb, 0, 1)
                # print(np.shape(temp_rgb))
                for j in range(0, 3):
                    processed_vid[j].append(temp_rgb[j])

            print(np.shape(processed_vid))
            data_hdf5 = dataset.create_dataset(name=str(i), data=processed_vid, shape=np.shape(processed_vid), dtype=int)
            data_hdf5.attrs['video'] = video_id
            data_hdf5.attrs['anomaly'] = 0

    def prepare_anomaly(self, config, dataset, anomaly_videos, normal_videos, raw_set):
        crop_size = config.prepare_crop_size
        #prepare_train
        #get 3000 anomaly samples from abnormal videos
        for i in range(0, 3000):
            video_id = anomaly_videos[np.random.randint(0, len(anomaly_videos), 1)[0]]

            l, r = raw_set[video_id][0], raw_set[video_id][1]
            anomaly_box = raw_set[video_id][2:6]
            print(anomaly_box)
            #find random abnormal interval
            lt = np.random.randint(l, r - config.prepare_len, 1)[0]
            rt = lt + config.prepare_len
            
            x1 = np.random.randint(max(0, anomaly_box[0] - crop_size[0]/2), anomaly_box[2] - crop_size[0]/2, 1)[0]
            y1 = np.random.randint(max(0, anomaly_box[1] - crop_size[1]/2), anomaly_box[3] - crop_size[1]/2, 1)[0]
            x2, y2 = np.array([x1, y1]) + np.array(crop_size)
            print('%d %d %d %d %d %d' % (lt, rt, x1, y1, x2, y2))

            #process video
            filename = self.root_dir + '/' + str(video_id) + '.mp4'
            vid = imageio.get_reader(filename, 'ffmpeg')
            processed_vid = [[], [], []]
            for frame_id in range(lt, rt, int(config.prepare_len / config.prepare_len_sample)):
                rgb = vid.get_data(frame_id)
                temp_rgb = rgb[y1:y2, x1:x2, :]
                temp_rgb = np.swapaxes(temp_rgb, 1, 2)
                temp_rgb = np.swapaxes(temp_rgb, 0, 1)
                #print(np.shape(temp_rgb))
                for j in range(0, 3):
                    processed_vid[j].append(temp_rgb[j])
            data_hdf5 = dataset.create_dataset(name=str(i), data=processed_vid, shape=np.shape(processed_vid), dtype=int)
            print(data_hdf5.name)
            data_hdf5.attrs['video'] = video_id
            data_hdf5.attrs['anomaly'] = 1

            #debug
            print(np.shape(processed_vid))
            fig = plt.figure()
            ims = []
            for ii in range(0, np.shape(processed_vid)[1]):
                im = plt.imshow(np.dstack((processed_vid[0][ii], processed_vid[1][ii], processed_vid[2][ii])), animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
            plt.show()
            
            print('finish')

    def prepare_data(self, config):

        # l r x1 y1 x2 y2
        raw_txt = self.root_dir + '/train-anomaly.txt'
        raw_set = {}
        with open(raw_txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_id, l, r, x1, y1, x2, y2 = [int(value) for value in line.split(' ')]
                raw_set[video_id] = [l, r, x1, y1, x2, y2]
        
        anomaly_videos = list(raw_set.keys())
        normal_videos = list(set(range(1, 101)) - set(anomaly_videos))

        #create dataset
        hdf5_dataset = h5py.File(config.prepare_hdf5_dir + '/AiCity.hdf5', 'w')
        train_set = hdf5_dataset.create_group('train')
        train_set.attrs['normal_length'] = 3000
        train_set.attrs['anomaly_length'] = 3000

        #self.prepare_normal(config, train_set, anomaly_videos, normal_videos, raw_set)
        self.prepare_anomaly(config, train_set, anomaly_videos, normal_videos, raw_set)

        hdf5_dataset.close()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from torch.utils.data import DataLoader
    import config_net
    train_data = VideoDataset(config_net, dataset='aicity', split='train', preprocess=True)    

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