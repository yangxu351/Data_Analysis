import os 
from PIL import Image
import numpy as np
import json 
import logging
logger = logging.getLogger(__name__)

def outlier_unreadable_img_detection(base_dir, outlier_thresh, abnormal_condition=0, split=None):
    '''
        outlier detection
    '''
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        img_dir = os.path.join(base_dir, sf, 'images')
        img_list = os.listdir(img_dir)

        outlier_list = []
        not_img_list = []
        for i_name in img_list:
            img_file = os.path.join(img_dir, i_name)
            try:
                img = np.array(Image.open(img_file))
            except:
                logging.warning(f'{img_file} is not readable!!')
                img = None
                not_img_list.append(img_file)
                continue
            # print(i_name, 'shape', img.shape) # h, w, c
            total_num = 1.
            for i in range(len(img.shape)):
                total_num *= img.shape[i]
            abnormal_ratio = np.count_nonzero(img==abnormal_condition)/(total_num*1.)
            if abnormal_ratio >= outlier_thresh:
                outlier_list.append(i_name) 
                
        with open(os.path.join(base_dir, f'{sf}_not_readable_imgs.txt'), 'w') as f:
            for img_f in not_img_list:
                f.write("%s\n" % (img_f))
        f.close()

        with open(os.path.join(base_dir, f'{sf}_outlier_imgs.txt'), 'w') as f:
            for img_f in outlier_list:
                f.write("%s\n" % (img_f))
        f.close()
    
    
def empty_unreadable_check_msk_png(base_dir, split=None):
    '''
        empty masks check
        unreadable masks check
    '''
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue

        msk_dir = os.path.join(base_dir, sf, 'masks')
        msk_list = os.listdir(msk_dir)

        empty_files = []
        unreadble_files = []
        for msk_name in msk_list:
            msk_file = os.path.join(msk_dir, msk_name)
            try:
                img = np.array(Image.open(msk_file))
            except:
                logger.warning(f'{msk_file} is not readable!!')
                img = None
                unreadble_files.append(msk_name)
                continue
            empty = np.all(img==0) 
            if empty:
                empty_files.append(msk_name)

        with open(os.path.join(base_dir, f'{sf}_empty_masks.txt'), 'w') as f:
            for msk_f in empty_files:
                f.write("%s\n" % msk_f)
        f.close()

        with open(os.path.join(base_dir, f'{sf}_unreadable_masks.txt'), 'w') as f:
            for msk_f in unreadble_files:
                f.write("%s\n" % msk_f)
        f.close()


def check_msk_id(base_dir, num_cls, split=None):
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        dict_invalid_msk_cid = {}
        msk_dir = os.path.join(base_dir, sf, 'masks')
        msk_list = os.listdir(msk_dir)

        for msk_name in msk_list:
            msk_file = os.path.join(msk_dir, msk_name)
            img = np.array(Image.open(msk_file))
            # if img[img<0]
            # if img.min()<0 or img.max()>=num_cls:
            #     invalid_msk_id_files.append(msk_name)
            outlier_values = np.unique(img[(img < 0) | (img >= num_cls)])
            if outlier_values.shape[0]:
                dict_invalid_msk_cid[msk_name] = outlier_values.tolist()
                    
        with open(os.path.join(base_dir, f'{sf}_invalid_msk_id_files.txt'), 'w') as f:
            json.dump(dict_invalid_msk_cid, f, ensure_ascii=False, indent=3)
        f.close()

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/Lunar'
    split = 'train'
    
    outlier_unreadable_img_detection(base_dir, split, outlier_thresh=0.5)

    empty_unreadable_check_msk_png(base_dir, split)
        
