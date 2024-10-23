import os 
from PIL import Image
import numpy as np
import pandas as pd

def outlier_unreadable_img_detection(base_dir, outlier_thresh=0.5):
    '''
        outlier detection
    '''
    split_folders = os.listdir(base_dir)
    for sf in split_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        if split in ['train', 'val', 'test']:

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
                    not_img_list.append(i_name)
                    continue
                # print(i_name, 'shape', img.shape) # h, w, c
                total_num = img.shape[0]*img.shape[1]*img.shape[2]
                non_zero_ratio = np.count_nonzero(img)/(total_num*1.)
                if non_zero_ratio >= outlier_thresh:
                    outlier_list.append(i_name) 
                    
            with open(os.path.join(base_dir, f'{split}_not_readable_imgs.txt'), 'w') as f:
                for name in not_img_list:
                    f.write("%s\n" % (name))
            f.close()

            with open(os.path.join(base_dir, f'{split}_outlier_imgs.txt'), 'w') as f:
                for name in outlier_list:
                    f.write("%s\n" % (name))
            f.close()
    

    
def empty_check_lbl(base_dir):
    '''
        empty label check
    '''
    split_folders = os.listdir(base_dir)
    for sf in split_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        if split in ['train', 'val', 'test']:

            lbl_dir = os.path.join(base_dir, split, 'labels')
            lbl_list = os.listdir(lbl_dir)
            empty_files = []
            for i_name in lbl_list:
                lbl_file = os.path.join(lbl_dir, i_name)
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    print("empty label file!")
                    empty_files.append(i_name)

            with open(os.path.join(base_dir, f'{split}_empty_files.txt'), 'w') as f:
                for name in empty_files:
                    f.write("%s\n" % (name))
            f.close()

def check_id(base_dir, num_cls):
    split_folders = os.listdir(base_dir)
    for sf in split_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        invalid_id_files = []
        if split in ['train', 'val', 'test']:
            lbl_dir = os.path.join(base_dir, split, 'labels')
            lbl_list = os.listdir(lbl_dir)
            for i_name in lbl_list:
                lbl_file = os.path.join(lbl_dir, i_name)
                df_lbl = pd.read_csv(lbl_file, delimiter='\t', header=None)
                if df_lbl.iloc[:, 0].max()>=num_cls or df_lbl.iloc[:, 0].min()<0:
                    invalid_id_files.append(i_name)

        with open(os.path.join(base_dir, f'{split}_invalid_id_files.txt'), 'w') as f:
            for name in invalid_id_files:
                f.write("%s\n" % (name))
        f.close()

def check_coordinates(base_dir, wh_ratio_thres=100):
    split_folders = os.listdir(base_dir)
    for sf in split_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        invalid_coor_files = []
        invalid_wh_ratio_files = []
        if split in ['train', 'val', 'test']:
            lbl_dir = os.path.join(base_dir, split, 'labels')
            lbl_list = os.listdir(lbl_dir)
            for i_name in lbl_list:
                lbl_file = os.path.join(lbl_dir, i_name)
                df_lbl = pd.read_csv(lbl_file, delimiter='\t', header=None)
                if df_lbl.iloc[:, 1:5].max()>1 or df_lbl.iloc[:, 1:5].min()<0:
                    print('invalid coordinates')
                    invalid_coor_files.append(i_name)
                w_d_h = df_lbl.iloc[:,3]/df_lbl.iloc[:,4]
                
                if w_d_h.min()<1./wh_ratio_thres or w_d_h.max()>wh_ratio_thres:
                    print('invalid wh ratio')
                    invalid_wh_ratio_files.append(i_name)

        with open(os.path.join(base_dir, f'{split}_invalid_coor_files.txt'), 'w') as f:
            for name in invalid_coor_files:
                f.write("%s\n" % (name))
        f.close()

        with open(os.path.join(base_dir, f'{split}_invalid_wh_ratio_files.txt'), 'w') as f:
            for name in invalid_wh_ratio_files:
                f.write("%s\n" % (name))
        f.close()

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/COCO'
    split = 'val'
    
    outlier_unreadable_img_detection(base_dir, split, outlier_thresh=0.5)

    empty_check_lbl(base_dir, split)
        
