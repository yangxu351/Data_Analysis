import os 
from PIL import Image
import numpy as np
import pandas as pd

def outlier_unreadable_img_detection(base_dir, outlier_thresh=0.5, split=None):
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
                print('not image')
                img = None
                not_img_list.append(i_name)
                continue
            # print(i_name, 'shape', img.shape) # h, w, c
            total_num = img.shape[0]*img.shape[1]*img.shape[2]
            non_zero_ratio = np.count_nonzero(img)/(total_num*1.)
            if 1-non_zero_ratio >= outlier_thresh:
                outlier_list.append(i_name) 
                
        with open(os.path.join(base_dir, f'{sf}_not_readable_imgs.txt'), 'w') as f:
            for name in not_img_list:
                f.write("%s\n" % (name))
        f.close()

        with open(os.path.join(base_dir, f'{sf}_outlier_imgs.txt'), 'w') as f:
            for name in outlier_list:
                f.write("%s\n" % (name))
        f.close()
    

    
def empty_check_lbl(base_dir, split=None):
    '''
        empty label check
    '''
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        lbl_dir = os.path.join(base_dir, sf, 'labels')
        lbl_list = os.listdir(lbl_dir)
        empty_files = []
        for i_name in lbl_list:
            lbl_file = os.path.join(lbl_dir, i_name)
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            if not lines:
                print("empty label file!")
                empty_files.append(i_name)

        with open(os.path.join(base_dir, f'{sf}_empty_files.txt'), 'w') as f:
            for name in empty_files:
                f.write("%s\n" % (name))
        f.close()
        

def empty_lbl_check_by_file(lbl_file):
    with open(lbl_file, 'r') as f:
        lines = f.readlines()
    if not lines:
        print("empty label file!")
        return True
    else:
        return False

def check_id(base_dir, num_cls, split=None):
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        invalid_id_files = []
        lbl_dir = os.path.join(base_dir, sf, 'labels')
        lbl_list = os.listdir(lbl_dir)
        for i_name in lbl_list:
            lbl_file = os.path.join(lbl_dir, i_name)
            if empty_lbl_check_by_file(lbl_file):
                continue
            df_lbl = pd.read_csv(lbl_file, delimiter=',', header=None)
            for ix in range(df_lbl.shape[0]):
                if df_lbl.iloc[ix, 0]>=num_cls or df_lbl.iloc[ix, 0]<0:
                    invalid_id_files.append((i_name, df_lbl.iloc[ix,0]))

        with open(os.path.join(base_dir, f'{sf}_invalid_id_files.txt'), 'w') as f:
            for name in invalid_id_files:
                f.write("%s,%d\n" % (name))
        f.close()

def check_coordinates(base_dir, wh_ratio_thres=10,split=None):
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        invalid_coor_files = []
        invalid_wh_ratio_files = []
        lbl_dir = os.path.join(base_dir, sf, 'labels')
        lbl_list = os.listdir(lbl_dir)
        for i_name in lbl_list:
            lbl_file = os.path.join(lbl_dir, i_name)
            if empty_lbl_check_by_file(lbl_file):
                continue
            df_lbl = pd.read_csv(lbl_file, delimiter=',', header=None)
            for ix in range(df_lbl.shape[0]):
                if df_lbl.iloc[ix, 1:5].max()>1 or df_lbl.iloc[ix, 1:5].min()<0:
                    print('invalid coordinates!')
                    invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
                # elif df_lbl.iloc[ix,1]+df_lbl.iloc[ix,3]/2. > 1 or df_lbl.iloc[ix,2]+df_lbl.iloc[ix,4]/2. > 1 or df_lbl.iloc[ix,1]-df_lbl.iloc[ix,3]/2.<0 or df_lbl.iloc[ix,2]-df_lbl.iloc[ix,4]/2.<0: 
                #     print('invalid coordinates!')
                #     invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
                elif df_lbl.iloc[ix,3]==0 or df_lbl.iloc[ix,4]==0:
                    print('invalid coordinates!')
                    invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))

                w_d_h = df_lbl.iloc[ix,3]/max(df_lbl.iloc[ix,4], 0.00001)
                if w_d_h.min()<1./wh_ratio_thres or w_d_h.max()>wh_ratio_thres:
                    print('invalid wh ratio')
                    invalid_wh_ratio_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))

        with open(os.path.join(base_dir, f'{sf}_invalid_coor_files.txt'), 'w') as f:
            for (name, bbx) in invalid_coor_files:
                f.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (name, bbx[0], bbx[1], bbx[2], bbx[3]))
        f.close()

        with open(os.path.join(base_dir, f'{sf}_invalid_wh_ratio_files.txt'), 'w') as f:
            for (name, bbx) in invalid_wh_ratio_files:
                f.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (name, bbx[0], bbx[1], bbx[2], bbx[3]))
        f.close()

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/COCO'
    split = 'val'
    
    outlier_unreadable_img_detection(base_dir, outlier_thresh=0.5, split=split)

    empty_check_lbl(base_dir, split)
        
