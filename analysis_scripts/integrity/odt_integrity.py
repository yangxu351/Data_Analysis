import os 
import numpy as np

def does_match_img_lbl(base_dir, split):
    img_dir = os.path.join(base_dir, split, 'images')
    lbl_dir = os.path.join(base_dir, split, 'labels')
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_name_wo_suffix = [f.split('.')[0] for f in img_list]
    img_wolbl_list = None 
    lbl_list = os.listdir(lbl_dir)
    lbl_list.sort()
    lbl_name_wo_suffix = [f.split('.')[0] for f in lbl_list]
    lbl_woimg_list = None 
    if len(lbl_list)>len(img_list):
        lbl_woimg_list = [os.path.join(lbl_dir, name) for name in lbl_list if name.split('.')[0] not in img_name_wo_suffix]
        arr_lbl_woimg = np.array(lbl_woimg_list)
        np.savetxt(os.path.join(base_dir, split, 'lbl_without_img.txt'), arr_lbl_woimg, delimiter='\n')

    elif len(lbl_list)<len(img_list):
        img_wolbl_list = [os.path.join(img_dir, name) for name in img_list if name.split('.')[0] not in lbl_name_wo_suffix]
        arr_img_wolbl = np.array(img_wolbl_list)
        np.savetxt(os.path.join(base_dir, split, 'img_without_lbl.txt'), arr_img_wolbl, delimiter='\n')

if __name__ == '__main__':
    base_dir = 'F:/Public_Dataset/COCO'
    split = 'val'
    does_match_img_lbl(base_dir, split)
    