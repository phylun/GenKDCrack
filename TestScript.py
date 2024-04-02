import numpy as np
import os
from shutil import copyfile

###### list files ######
data_root = './dataset_sample/unlabeled/trainConc/'
img_folder = 'JPEGImages'
files = [f[:-4] for f in os.listdir(os.path.join(data_root, img_folder)) if f.find('png') > -1]
print(len(files))
with open(os.path.join(data_root, 'gentrainConc.txt'), 'w') as f:
    for file in files:
        str_file = file + '\n'
        f.write(str_file)


###### copy files of unlabeled ######
# src_data = 'E:\\MyProject\\Data\\CrackSegmentData_gen_origin\\trainConc'
# img_folder = 'JPEGImages'
# dst_data = './dataset_sample/unlabeled/trainConc'

# files = [f[:-4] for f in os.listdir(os.path.join(src_data, img_folder)) if f.find('png') > -1]
# np.random.seed(1234)
# np.random.shuffle(files)
# print(len(files))
# for file in files[:300]:
#     src_img_file = os.path.join(src_data, img_folder, file + '.png')        
#     dst_img_file = os.path.join(dst_data, img_folder, file + '.png')    
    
#     copyfile(src_img_file, dst_img_file)    


###### copy files of labeled ######
# src_data = 'E:\\MyProject\\Data\\CrackSegmentData_clean\\testConc'
# img_folder = 'JPEGImages'
# lbl_folder = 'SegmentationClass'
# dst_data = './dataset_sample/labeled/testConc'

# files = [f[:-4] for f in os.listdir(os.path.join(src_data, img_folder)) if f.find('jpg') > -1]
# np.random.seed(1234)
# np.random.shuffle(files)
# print(len(files))
# for file in files[:50]:
#     src_img_file = os.path.join(src_data, img_folder, file + '.jpg')
#     src_lbl_file = os.path.join(src_data, lbl_folder, file + '.png')
    
#     dst_img_file = os.path.join(dst_data, img_folder, file + '.jpg')
#     dst_lbl_file = os.path.join(dst_data, lbl_folder, file + '.png')
    
#     copyfile(src_img_file, dst_img_file)
#     copyfile(src_lbl_file, dst_lbl_file)
    
    
    

