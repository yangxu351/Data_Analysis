import os 
from PIL import Image
import numpy as np

def outlier_img_detection(base_dir, split, outlier_thresh=0.5):
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
    if not_img_list:
        arr_not_imgs = np.array(not_img_list)
        np.savetxt(os.path.join(base_dir, split, 'not_readable_imgs.txt'), arr_not_imgs, delimiter='\n')
    if outlier_list:
        arr_outlier_imgs = np.array(outlier_list)
        np.savetxt(os.path.join(base_dir, split, 'outlier_imgs.txt'), arr_outlier_imgs, delimiter='\n')


    
def empty_check_lbl(base_dir, split):
    '''
        empty label check
    '''
    lbl_dir = os.path.join(base_dir, split, 'labels')
    lbl_list = os.listdir(lbl_dir)

    empty_files = []
    for lbl_name in lbl_list:
        lbl_file = os.path.join(lbl_dir, lbl_name)
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
        if not lines:
            print("label file is empty!")
            empty_files.append(f)

    arr_empty_file = np.array(empty_files)
    np.savetxt(os.path.join(base_dir, split, 'empty_lbls.txt'), arr_empty_file, delimiter='\n')

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/COCO'
    split = 'val'
    
    outlier_img_detection(base_dir, split, outlier_thresh=0.5)

    empty_check_lbl(base_dir, split)
        
