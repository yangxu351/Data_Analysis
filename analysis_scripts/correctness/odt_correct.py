import os 
from PIL import Image
import numpy as np

def outlier_unreadable_img_detection(base_dir, split, outlier_thresh=0.5):
    '''
        outlier detection
    '''
    img_dir = os.path.join(base_dir, split, 'images')
    img_list = os.listdir(img_dir)

    outlier_list = []
    not_img_list = []
    for i_name in img_list:
        img_file = os.path.join(img_dir, i_name)
        try:
            img = np.array(Image.open(img_file))
        except:
            print('not image')
            img = None
            not_img_list.append(img_file)
            continue
        # print(i_name, 'shape', img.shape) # h, w, c
        total_num = img.shape[0]*img.shape[1]*img.shape[2]
        non_zero_ratio = np.count_nonzero(img)/(total_num*1.)
        if non_zero_ratio >= outlier_thresh:
            outlier_list.append(img_file) 
            
    with open(os.path.join(base_dir, f'{split}_not_readable_imgs.txt'), 'w') as f:
        for name in not_img_list:
            f.write("%s\n" % (name))
    f.close()

    with open(os.path.join(base_dir, f'{split}_outlier_imgs.txt'), 'w') as f:
        for name in outlier_list:
            f.write("%s\n" % (name))
    f.close()
    

    
def empty_check_lbl(base_dir, split):
    '''
        empty label check
    '''
    lbl_dir = os.path.join(base_dir, split, 'labels')
    lbl_list = os.listdir(lbl_dir)
    empty_files = []
    for i_name in lbl_list:
        lbl_file = os.path.join(lbl_dir, i_name)
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
        if not lines:
            print("文件为空。")
            empty_files.append(lbl_file)

    with open(os.path.join(base_dir, f'{split}_empty_files.txt'), 'w') as f:
        for name in empty_files:
            f.write("%s\n" % (name))
    f.close()

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/COCO'
    split = 'val'
    
    outlier_unreadable_img_detection(base_dir, split, outlier_thresh=0.5)

    empty_check_lbl(base_dir, split)
        
