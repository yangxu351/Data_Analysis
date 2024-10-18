import os 
from PIL import Image
import numpy as np

def outlier_unreadable_img_detection(base_dir, split, outlier_thresh, abnormal_condition=0):
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
        abnormal_ratio = np.count_nonzero(img==abnormal_condition)/(total_num*1.)
        if abnormal_ratio >= outlier_thresh:
            outlier_list.append(img_file) 
            
    with open(os.path.join(base_dir, f'{split}_not_readable_imgs.txt'), 'w') as f:
        for name in not_img_list:
            f.write("%s\n" % (name))
    f.close()

    with open(os.path.join(base_dir, f'{split}_outlier_imgs.txt'), 'w') as f:
        for name in outlier_list:
            f.write("%s\n" % (name))
    f.close()
    
    
def empty_unreadable_check_msk_png(base_dir, split):
    '''
        empty masks check
        unreadable masks check
    '''
    msk_dir = os.path.join(base_dir, split, 'masks')
    msk_list = os.listdir(msk_dir)

    empty_files = []
    unreadble_files = []
    for msk_name in msk_list:
        msk_file = os.path.join(msk_dir, msk_name)
        try:
            img = np.array(Image.open(msk_file))
        except:
            print('not image')
            img = None
            unreadble_files.append(msk_file)
            continue
        empty = np.all(img==0) 
        if empty:
            empty_files.append(msk_file)

    with open(os.path.join(base_dir, f'{split}_empty_masks.txt'), 'w') as f:
        for name in empty_files:
            f.write("%s\n" % name)
    f.close()

    with open(os.path.join(base_dir, f'{split}_unreadable_masks.txt'), 'w') as f:
        for name in unreadble_files:
            f.write("%s\n" % name)
    f.close()

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/Lunar'
    split = 'train'
    
    outlier_unreadable_img_detection(base_dir, split, outlier_thresh=0.5)

    empty_unreadable_check_msk_png(base_dir, split)
        
