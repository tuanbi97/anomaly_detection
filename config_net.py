#prepare data
def db_dir(dataset = 'aicity'):
    if (dataset == 'aicity'):
        root_dir = '/home/tuanbi97/anomaly_detection/data'
        output_dir = '/home/tuanbi97/anomaly_detection/output'
    
    return root_dir, output_dir

dataname = 'AiCityTest.hdf5'
test_normal_size = 100
test_anomaly_size = 100
test_aug_anomaly_size = 0

prepare_cropped_vehicles = '/home/tuanbi97/anomaly_detection/data/cropped_vehicles'
prepare_train_dir = '/home/tuanbi97/anomaly_detection/data/train'
prepare_test_dir = '/home/tuanbi97/anomaly_detection/data/test'
prepare_hdf5_dir = '/home/tuanbi97/anomaly_detection/data'
prepare_crop_size = [160, 160]
prepare_len = 150
prepare_len_sample = 30
prepare_video_max_len = 26000
video_size = [800, 410]
video_fps = 30
resize_w = 160
resize_h = 160
#resized_shape = [171, 128]
resized_shape = (112, 112)
crop_size = 112

#train
batch_size = 2
step_per_epoch = 2
step_w = 80
step_h = 125


#test
test_dir='/home/tuanbi97/anomaly_detection/data'
test_video_step = 150
test_frame_step = [160, 125]
#model_dir='run/run_10/models/R2Plus1D-aicity_epoch-79.pth.tar'
model_dir='models/R2Plus1D-aicity_epoch-15.pth.tar'
