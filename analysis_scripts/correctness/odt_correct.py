import os 
from PIL import Image
import numpy as np
import pandas as pd
import json 
import logging
logger = logging.getLogger(__name__)


def outlier_unreadable_img_detection(base_dir, dict_stat, outlier_thresh=0.5, split=None):
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
                logger.warning(f'{i_name} is not readable!!')
                img = None
                not_img_list.append(i_name)
                continue
            # print(i_name, 'shape', img.shape) # h, w, c
            total_num = 1.
            for i in range(len(img.shape)):
                total_num *= img.shape[i]
            non_zero_ratio = np.count_nonzero(img)/total_num*1
            if 1-non_zero_ratio >= outlier_thresh:
                outlier_list.append(i_name) 
                
        with open(os.path.join(base_dir, f'{sf}_damaged_imgs.txt'), 'w') as f:
            for name in not_img_list:
                f.write("%s\n" % (name))
        f.close()
        dict_stat['图像损坏率'] = f'{round(len(not_img_list)/len(img_list),4)*100}%'

        with open(os.path.join(base_dir, f'{sf}_outlier_imgs.txt'), 'w') as f:
            for name in outlier_list:
                f.write("%s\n" % (name))
        f.close()
        dict_stat['图像异常比例'] = f'{round(len(outlier_list)/len(img_list),4)*100}%'


# def empty_check_lbl(base_dir, dict_stat, split=None):
#     '''
#         empty label check
#     '''
#     if split:
#         valid_folders = [split]
#     else:
#         valid_folders = ['train', 'val', 'test']
#     for sf in valid_folders:
#         if not os.path.isdir(os.path.join(base_dir, sf)):
#             continue
#         lbl_dir = os.path.join(base_dir, sf, 'labels')
#         lbl_list = os.listdir(lbl_dir)
#         empty_files = []
#         for i_name in lbl_list:
#             lbl_file = os.path.join(lbl_dir, i_name)
#             with open(lbl_file, 'r') as f:
#                 lines = f.readlines()
#             if not lines:
#                 logger.warning(f"{i_name}, empty label file!")
#                 empty_files.append(i_name)

#         with open(os.path.join(base_dir, f'{sf}_yolo_empty_files.txt'), 'w') as f:
#             for name in empty_files:
#                 f.write("%s\n" % (name))
#         f.close()
#         dict_stat['空标注比例'] = f'{round(len(empty_files)/len(lbl_list),4)*100}%'
        

def empty_lbl_check_by_file(lbl_file):
    with open(lbl_file, 'r') as f:
        lines = f.readlines()
    if not lines:
        return True
    else:
        return False

# def check_id_readable(base_dir, dict_stat, mapping_file, split=None):
#     if split:
#         valid_folders = [split]
#     else:
#         valid_folders = ['train', 'val', 'test']
    
#     dict_id_cat = json.load(open(mapping_file))
#     classes = [x for x in dict_id_cat.values()]
#     # ids = [x for x in dict_id_cat.keys()]
#     # dict_cat_id = {classes[ix]:ids[ix] for ix in range(len(classes))}
#     num_cls = len(classes)

#     for sf in valid_folders:
#         if not os.path.isdir(os.path.join(base_dir, sf)):
#             continue
#         invalid_id_files = []
#         unreadable_files = []
#         lbl_dir = os.path.join(base_dir, sf, 'labels')
#         lbl_list = os.listdir(lbl_dir)
#         for i_name in lbl_list:
#             lbl_file = os.path.join(lbl_dir, i_name)
#             try:
#                 df_lbl = pd.read_csv(lbl_file, delimiter=' ', header=None)
#             except:
#                 logger.error(f'{i_name} is unreadable!!')
#                 unreadable_files.append(i_name)
#                 continue 
#             if empty_lbl_check_by_file(lbl_file):
#                 logger.warning(f"{i_name} empty label file!")
#                 continue
           
#             for ix in range(df_lbl.shape[0]):
#                 if df_lbl.iloc[ix, 0]>=num_cls or df_lbl.iloc[ix, 0]<0:
#                     invalid_id_files.append((i_name, df_lbl.iloc[ix,0]))

#         with open(os.path.join(base_dir, f'{sf}_yolo_unreadable_files.txt'), 'w') as f:
#             for name in unreadable_files:
#                 f.write("%s\n" % (name))
#         f.close()
#         dict_stat['标注损坏率'] =  f'{round(len(unreadable_files)/len(lbl_list),4)*100}%'

#         with open(os.path.join(base_dir, f'{sf}_yolo_invalid_id_files.txt'), 'w') as f:
#             for name in invalid_id_files:
#                 f.write("%s,%d\n" % (name))
#         f.close()
#         dict_stat['标注类别错误率'] =  f'{round(len(invalid_id_files)/len(lbl_list),4)*100}%'

# def check_coordinates(base_dir, dict_stat, wh_ratio_thres=10,split=None):
#     if split:
#         valid_folders = [split]
#     else:
#         valid_folders = ['train', 'val', 'test']
#     for sf in valid_folders:
#         if not os.path.isdir(os.path.join(base_dir, sf)):
#             continue
#         invalid_coor_files = []
#         invalid_wh_ratio_files = []
#         uncomplete_files = []
#         duplicate_files = []
#         lbl_dir = os.path.join(base_dir, sf, 'labels')
#         lbl_list = os.listdir(lbl_dir)
#         for i_name in lbl_list:
#             lbl_file = os.path.join(lbl_dir, i_name)
#             if empty_lbl_check_by_file(lbl_file):
#                 continue
#             df_lbl = pd.read_csv(lbl_file, delimiter=' ', header=None)
#             if df_lbl.shape[1] < 5:
#                 logger.error(f'{i_name} is uncomplete!')
#                 uncomplete_files.append(i_name)
#                 continue
#             has_duplicate = df_lbl.duplicated().any()
#             if has_duplicate:
#                 duplicate_files.append(i_name)
#                 logger.warning(f'{i_name} contains duplicate bbox!')

#             for ix in range(df_lbl.shape[0]):
#                 if df_lbl.iloc[ix, 1:5].max()>1 or df_lbl.iloc[ix, 1:5].min()<0:
#                     logger.warning(f'{i_name} contains invalid coordinates!')
#                     invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
#                 # elif df_lbl.iloc[ix,1]+df_lbl.iloc[ix,3]/2. > 1 or df_lbl.iloc[ix,2]+df_lbl.iloc[ix,4]/2. > 1 or df_lbl.iloc[ix,1]-df_lbl.iloc[ix,3]/2.<0 or df_lbl.iloc[ix,2]-df_lbl.iloc[ix,4]/2.<0: 
#                 #     print('invalid coordinates!')
#                 #     invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
#                 elif df_lbl.iloc[ix,3]<=0 or df_lbl.iloc[ix,4]<=0:
#                     logger.warning(f'{i_name} contains invalid coordinates!')
#                     invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))

#                 w_d_h = df_lbl.iloc[ix,3]/max(df_lbl.iloc[ix,4], 0.00001)
#                 if w_d_h.min()<1./wh_ratio_thres or w_d_h.max()>wh_ratio_thres:
#                     logger.warning(f'{i_name} contains invalid wh ratio')
#                     invalid_wh_ratio_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
        
#         with open(os.path.join(base_dir, f'{sf}_yolo_duplicate_files.txt'), 'w') as f:
#             for name in duplicate_files:
#                 f.write("%s\n" % (name))
#         f.close()
#         dict_stat['标注重复率'] = f'{round(len(duplicate_files)/len(lbl_list),4)*100}%' 

#         with open(os.path.join(base_dir, f'{sf}_invalid_coor_files.txt'), 'w') as f:
#             for (name, bbx) in invalid_coor_files:
#                 f.write("%s %.4f %.4f %.4f %.4f\n" % (name, bbx[0], bbx[1], bbx[2], bbx[3]))
#         f.close()
#         dict_stat['标注取值异常比例'] = f'{round(len(invalid_coor_files)/len(lbl_list),4)*100}%' 

#         with open(os.path.join(base_dir, f'{sf}_yolo_invalid_wh_ratio_files.txt'), 'w') as f:
#             for (name, bbx) in invalid_wh_ratio_files:
#                 f.write("%s %.4f %.4f %.4f %.4f\n" % (name, bbx[0], bbx[1], bbx[2], bbx[3]))
#         f.close()
#         dict_stat['标注宽高比失调率'] = f'{round(len(invalid_wh_ratio_files)/len(lbl_list),4)*100}%' 

#         with open(os.path.join(base_dir, f'{sf}_uncomplete_coor_files.txt'), 'w') as f:
#             for name in uncomplete_files:
#                 f.write("%s\n" % (name))
#         f.close()
#         dict_stat['标注关键字段缺失比例'] = f'{round(len(uncomplete_files)/len(lbl_list),4)*100}%' 

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/COCO'
    split = 'val'
    
    outlier_unreadable_img_detection(base_dir, outlier_thresh=0.5, split=split)

    # empty_check_lbl(base_dir, split)
        
