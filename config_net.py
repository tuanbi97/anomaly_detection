#prepare data
def db_dir(dataset = 'aicity'):
    if (dataset == 'aicity'):
        root_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data'
        output_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/output'
    
    return root_dir, output_dir

prepare_train_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data/train'
prepare_test_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data/test'
prepare_hdf5_dir = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/VestyCity/data'
prepare_crop_size = [160, 160]
prepare_len = 150
prepare_len_sample = 30
prepare_video_max_len = 26000
video_size = [800, 410]
video_fps = 30
# resize_w = 160
# resize_h = 160
resize_h = 128
resize_w = 171
crop_size = 112


#test
test_dir='/media/tuanbi97/Vesty/Datasets/aic19-track3-test-data/'
test_video_step = 150
test_frame_step = [160, 125]