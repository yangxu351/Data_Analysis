import os 
import sys 
from PIL import Image
import numpy as np
import json
import logging
logger = logging.getLogger(__name__)

def outlier_unreadable_img_detection(base_dir, split_list, outlier_thresh, mapping_file=None, abnormal_condition=0):
    '''
        outlier detection
        unreadable detection
    '''
    if mapping_file is None:
        logging.error('class mapping file is needed!')
        sys.exit(1)
    dict_id_cat = json.load(open(mapping_file))
    ori_cats = list(dict_id_cat.values())

    for sf in split_list:
        img_dir = os.path.join(base_dir, 'inputs', f'{sf}_dataset')
        cat_list = os.listdir(img_dir)
        abnormal_cat_list = []
        outlier_list = []
        not_img_list = []
        for cat in cat_list:
            if cat not in ori_cats:
                logger.warning(f'{cat} not in category list!')
                abnormal_cat_list.append(cat)

            file_list = os.listdir(os.path.join(img_dir, cat))
            for img_name in file_list:
                img_file = os.path.join(img_dir, cat, img_name)
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
                abnormal_ratio = np.count_nonzero(img==abnormal_condition)/total_num
                if abnormal_ratio >= outlier_thresh:
                    outlier_list.append(img_file) 
        
        with open(os.path.join(base_dir, f'{split}_unrecognized_categories.txt'), 'w') as f:
            for name in abnormal_cat_list:
                f.write("%s\n" % (name))
        f.close()

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

        
