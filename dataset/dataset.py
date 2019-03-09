import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

class VideoDataset(Dataset):
    def __init__(self, config, dataset='aicity', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = config.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if preprocess == True:
            self.prepare_data(config)

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

        crop_size = config.prepare_crop_size

        #prepare_train
        for i in range(0, 2000):
            video_id = anomaly_videos[np.random.randint(0, len(anomaly_videos), 1)[0]]

            l, r = raw_set[video_id][0], raw_set[video_id][1]
            for j in range(0, 100):
                lt = np.random.randint(0, config.prepare_video_max_len, 1)[0]
                if lt >= l:
                    lt +=  r - l
                rt = lt + config.prepare_len
                if rt > config.prepare_video_max_len:
                    continue
                if lt < l and rt > l and 1.0 * (rt - l) / config.prepare_len >= 0.5:
                    continue
                if rt > r and lt < r and 1.0 * (r - lt) / config.prepare_len >= 0.5:
                    continue
                if l <= lt and rt <= r:
                    continue
                break
            
            x1 = np.random.randint(0, config.video_size[0] - config.prepare_crop_size[0], 1)[0]
            y1 = np.random.randint(0, config.video_size[1] - config.prepare_crop_size[1], 1)[0]
            print(config.prepare_crop_size)
            x2, y2 = np.array([x1, y1]) + np.array(config.prepare_crop_size)
            print('%d %d %d %d %d %d' % (lt, rt, x1, y1, x2, y2))

            #process video
            filename = self.root_dir + '/' + str(video_id) + '.mp4'
            vid = imageio.get_reader(filename, 'ffmpeg')
            processed_vid = [[], [], []]
            for frame_id in range(lt, rt):
                rgb = vid.get_data(frame_id)
                temp_rgb = rgb[y1:y2, x1:x2, :]
                temp_rgb = np.swapaxes(temp_rgb, 1, 2)
                temp_rgb = np.swapaxes(temp_rgb, 0, 1)
                print(np.shape(temp_rgb))
                for j in range(0, 3):
                    processed_vid[j].append(temp_rgb[j])
                # for j in range(0, 3):
                #     plt.imshow(temp_rgb[j])
                #     plt.show()

            #debug
            print(np.shape(processed_vid))
            fig = plt.figure()
            ims = []
            for i in range(0, np.shape(processed_vid)[1], 5):
                im = plt.imshow(np.dstack((processed_vid[0][i], processed_vid[1][i], processed_vid[2][i])), animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
            plt.show()
            
            print('finish')

        for i in range(0, 3000):
            normal_videos = normal_videos[i]
            

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from torch.utils.data import DataLoader
    import config_net
    train_data = VideoDataset(config_net, dataset='aicity', split='train', clip_len=15, preprocess=True)    

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