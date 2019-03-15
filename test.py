from network import R2Plus1D_model
import torch
import config_net as config
import imageio
import cv2
import numpy as np

video_id = '1.mp4'

filename = config.test_dir + video_id
capture = cv2.VideoCapture(filename)
video_max_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
vid = imageio.get_reader(filename, 'ffmpeg')
test_start = 0
test_end = video_max_len

for frame_start in range(test_start, test_end, config.test_video_step):
    video_segment = []
    for i in range(0, 15):
        video_segment.append([])
    for i in range(0, 150, 5):
        rgb = vid.get_data(frame_start + i)
        segment_id = 0
        for x1 in range(0, config.video_size[0], config.test_frame_step[0]):
            for y1 in range(0, config.video_size[1], config.test_frame_step[1]):
                x2 = x1 + config.prepare_crop_size[0]
                y2 = y1 + config.prepare_crop_size[1]
                if (x2 > config.video_size[0] or y2 > config.video_size[1]):
                    break
                print(x1, ' ', y1,' ', x2,' ', y2)
                temp_rgb = rgb[y1: y2, x1: x2, :]
                temp_rgb = cv2.resize(temp_rgb, (config.crop_size, config.crop_size))
                temp_rgb = np.swapaxes(temp_rgb, 1, 2)
                temp_rgb = np.swapaxes(temp_rgb, 0, 1)
                video_segment[segment_id].append(temp_rgb)
                segment_id += 1

    video_segment = np.array(video_segment)
    video_segment = np.swapaxes(video_segment, 1, 2)
    print(np.shape(video_segment))
    break

# inputs = torch.rand(1, 3, 16, 112, 112)
# net = R2Plus1D_model.R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False)

# outputs = net.forward(inputs)
# print(outputs.size())


