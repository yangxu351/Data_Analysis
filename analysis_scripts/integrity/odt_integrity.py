import os 
import numpy as np

def does_match_img_lbl(base_dir, split=None):
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        img_dir = os.path.join(base_dir, sf, 'images')
        lbl_dir = os.path.join(base_dir, sf, 'labels')
        img_list = os.listdir(img_dir)
        img_list.sort()
        img_name_wo_suffix = [f.split('.')[0] for f in img_list]
        img_wolbl_list = None 
        lbl_list = os.listdir(lbl_dir)
        lbl_list.sort()
        lbl_name_wo_suffix = [f.split('.')[0] for f in lbl_list]
        lbl_woimg_list = None 
        if len(lbl_list)>len(img_list):
            print('labels more than images!!!')
            lbl_woimg_list = [os.path.join(lbl_dir, name) for name in lbl_list if name.split('.')[0] not in img_name_wo_suffix]
            arr_lbl_woimg = np.array(lbl_woimg_list)
            np.savetxt(os.path.join(base_dir, f'{sf}_lbl_without_img.csv'), arr_lbl_woimg, fmt='%s')

        elif len(lbl_list)<len(img_list):
            print('images more than labels!!!')
            img_wolbl_list = [os.path.join(img_dir, name) for name in img_list if name.split('.')[0] not in lbl_name_wo_suffix]
            arr_img_wolbl = np.array(img_wolbl_list)
            np.savetxt(os.path.join(base_dir, f'{sf}_img_without_lbl.csv'), arr_img_wolbl, fmt='%s')

if __name__ == '__main__':
    base_dir = 'F:/Public_Dataset/COCO'
    split = 'val'
    does_match_img_lbl(base_dir)
    