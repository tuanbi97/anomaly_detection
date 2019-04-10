#prepare data
def db_dir(dataset = 'aicity'):
    if (dataset == 'aicity'):
        root_dir = '/home/vhvkhoa/aic_track3/anomaly_detection/data/aic19-track3-train-data'
        output_dir = '/home/vhvkhoa/aic_track3/anomaly_detection/output'

    return root_dir, output_dir

dataname = 'AiCity.hdf5'
prepare_cropped_vehicles = '/home/vhvkhoa/aic_track3/anomaly_detection/data/aic19-track3-train-data/cropped_vehicles'
prepare_train_dir = '/home/vhvkhoa/aic_track3/aic19-track3-train-data'
prepare_test_dir = '/home/vhvkhoa/aic_track3/aic19-track3-test-data'
prepare_hdf5_dir = '/home/vhvkhoa/aic_track3/anomaly_detection/data/aic19-track3-train-data'
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
step_per_epoch = 10
step_w = 80
step_h = 125


#test
test_dir = '/home/vhvkhoa/aic_track3/aic19-track3-train-data/'
test_video_step = 150
test_frame_step = [160, 125]
model_dir='models/R2Plus1D-aicity_epoch-33.pth.tar'
