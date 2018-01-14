small_width = 400
small_height = 400

project_name='ISIC-2017_Orig_train_data'
data_folder  ='/home/ramesh/Data/isic2017_data/'+project_name+'/'
target_train_dir = data_folder + 'isic2017_laasya_data/'
target_train_dir_aug = data_folder + 'isic2017_laasya_data_aug/'
gt_file = 'ISIC-2017_Training_Part3_GroundTruth_final.csv'
gt_nb_classes = 3
lb_fit_array = [0.0, 1.0, 2.0]
sCheckData = 'ISIC'
'''
project_name='family_faces'
data_folder  ='/home/ramesh/Data/'+project_name+'/'
target_train_dir = data_folder
gt_file = 'famiy_GroundTruth.csv'
gt_nb_classes = 4
lb_fit_array = [0, 1, 2, 3]
sCheckData = ''
'''
target_img_ext = '.jpg'
target_img_width = 299
target_img_height = 299
