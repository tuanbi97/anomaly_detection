#prepare data
def db_dir(dataset = 'aicity'):
    if (dataset == 'aicity'):
        root_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data2'
        output_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/output'
    
    return root_dir, output_dir

dataname = 'AiCity.hdf5'
prepare_cropped_vehicles = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data/cropped_vehicles'
prepare_train_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data/train'
prepare_test_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data/test'
prepare_hdf5_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data'
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
step_per_epoch = 1
step_w = 80
step_h = 125


#test
test_dir='/media/tuanbi97/Vesty/Datasets/aic19-track3-train-data/'
test_video_step = 150
test_frame_step = [160, 125]
#model_dir='run/run_10/models/R2Plus1D-aicity_epoch-79.pth.tar'
model_dir='models/R2Plus1D-aicity_epoch-15.pth.tar'