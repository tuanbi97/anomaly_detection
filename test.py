from network import R2Plus1D_model
import torch
import config_net as config
import imageio
import cv2
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    '--test-dir',
    dest='test_dir',
    help='directory for visualization pdfs (default: /tmp/infer_simple)',
    default='./../aic19-track3-train-data/',
    type=str
)
parser.add_argument(
    '--model-dir',
    dest='model_dir',
    help='directory for visualization pdfs (default: /tmp/infer_simple)',
    default='./../Ex1/R2Plus1D-aicity_epoch-33.pth.tar',
    type=str
)

args = parser.parse_args()
config.model_dir = args.model_dir
config.test_dir = args.test_dir
print config.model_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = R2Plus1D_model.R2Plus1DClassifier(2, (2, 2, 2, 2), pretrained=True).to(device)

video_id = '93.mp4'

filename = config.test_dir + video_id
capture = cv2.VideoCapture(filename)
video_max_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
capture.release()
vid = imageio.get_reader(filename, 'ffmpeg')
test_start = 4 * 60 * config.video_fps
test_end = video_max_len

anomaly_graph = []
f = open('check_video_2.txt', 'w')

start_time = time.time()
for frame_start in range(test_start, test_end, config.test_video_step):
    if (frame_start + config.prepare_len > test_end):
        break
    frame_start_time = time.time()
    video_segment = []
    for i in range(0, 15):
        video_segment.append([])
    for i in range(0, config.prepare_len, config.prepare_len//config.prepare_len_sample):
        rgb = vid.get_data(frame_start + i)

        segment_id = 0
        for x1 in range(0, config.video_size[0], config.test_frame_step[0]):
            for y1 in range(0, config.video_size[1], config.test_frame_step[1]):
                x2 = x1 + config.prepare_crop_size[0]
                y2 = y1 + config.prepare_crop_size[1]
                if (x2 > config.video_size[0] or y2 > config.video_size[1]):
                    break
                #print(x1, ' ', y1,' ', x2,' ', y2)
                temp_rgb = rgb[y1: y2, x1: x2, :]
                temp_rgb = cv2.resize(temp_rgb, (config.crop_size, config.crop_size))
                temp_rgb = np.swapaxes(temp_rgb, 1, 2)
                temp_rgb = np.swapaxes(temp_rgb, 0, 1)

                video_segment[segment_id].append(temp_rgb)
                segment_id += 1

    video_segment = np.array(video_segment, dtype=np.float32)
    video_segment = np.swapaxes(video_segment, 1, 2)
    print(np.shape(video_segment))
    #debug
    for i in range(0, len(video_segment)):
        processed_vid = video_segment[i].astype(int)
        fig = plt.figure()
        ims = []
        for ii in range(0, np.shape(processed_vid)[1]):
            im = plt.imshow(np.dstack((processed_vid[0][ii], processed_vid[1][ii], processed_vid[2][ii])), animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
        plt.show()
    # inputs = Variable(torch.from_numpy(video_segment), requires_grad=True).to(device)
    # outputs = net(inputs)
    # print(outputs.size())
    # break
    c_potential = 0
    s_potential = 0
    outputs = []

    xticks = [x for x in range(0, config.video_size[0] - config.prepare_crop_size[0] + 1, config.test_frame_step[0])]
    yticks = [y for y in range(0, config.video_size[1] - config.prepare_crop_size[1] + 1, config.test_frame_step[1])]
    hm = np.zeros([3, 5])
    for i in range(0, 15):
        inputs = Variable(torch.from_numpy(np.array([video_segment[i]], dtype=np.float32)), requires_grad=False).to(device)
        output = net(inputs)
        probs = torch.nn.Softmax(dim=1)(output)
        output = probs.cpu().detach().numpy()[0]
        outputs.append(output)
        hm[i % 3][i // 3] = output[1]

    #show heat map
    fig, ax = plt.subplots()
    im = ax.imshow(hm)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    plt.show()

    for i in range(0, 15):
        if outputs[i][1] > 0.7:
            s_potential += outputs[i][1]
            c_potential += 1
    
    if (c_potential > 0):
        anomaly_graph.append(s_potential/c_potential)
    else:
        anomaly_graph.append(0)

    f.write(str(anomaly_graph[-1]) + '\n')

    print(anomaly_graph[-1])
    print('frame finish time: ', time.time() - frame_start_time)
    print(outputs)

duration = time.time() - start_time    
print('execution time:', duration)
f.close()
plt.plot(anomaly_graph)
plt.show()

# inputs = torch.rand(1, 3, 16, 112, 112)
# net = R2Plus1D_model.R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False)

# outputs = net.forward(inputs)
# print(outputs.size())
