import os 
from PIL import Image
import numpy as np

def outlier_unreadable_img_detection(base_dir, outlier_thresh, abnormal_condition=0):
    '''
        outlier detection
        unreadable detection
    '''
    split_folders = os.listdir(base_dir)
    for sf in split_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        if split in ['train', 'val', 'test']:
            img_dir = os.path.join(base_dir, split)
            cat_list = os.listdir(img_dir)

            outlier_list = []
            not_img_list = []
            for cat in cat_list:
                file_list = os.listdir(os.path.join(img_dir, cat))
                for img_name in file_list:
                    img_file = os.path.join(img_dir, cat, img_name)
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

    

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/MSTAR'
    split = 'test'
    outlier_thresh = 0.5
    abnormal_condition = 255
    outlier_unreadable_img_detection(base_dir, split, outlier_thresh, abnormal_condition)

        
