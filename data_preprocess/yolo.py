import os 
import numpy as np
import pandas as pd
import json 
import logging
logger = logging.getLogger(__name__)


def dose_match_img_lbl(base_dir, dict_stat):
    lbl_dir = os.path.join(base_dir, 'labels')
    img_dir = os.path.join(base_dir, 'images')
    
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_name_wo_suffix = [f.split('.')[0] for f in img_list]
    img_wolbl_list = None 

    lbl_list = os.listdir(lbl_dir)
    lbl_list.sort()
    lbl_name_wo_suffix = [f.split('.')[0] for f in lbl_list]
    lbl_woimg_list = None 

    file_names = [x for x in img_name_wo_suffix if x in lbl_name_wo_suffix]
    if len(lbl_list)>len(file_names):
        logger.warning('labels more than images!!!')
        lbl_woimg_list = [name for name in lbl_list if name.split('.')[0] not in img_name_wo_suffix]
        arr_lbl_woimg = np.array(lbl_woimg_list)
        np.savetxt(os.path.join(base_dir, 'dismatch_lbl_without_img.csv'), arr_lbl_woimg, fmt='%s')
        dict_stat['图像标注不匹配率'] = f'{round(len(lbl_woimg_list)/len(lbl_list),4)*100}%'
    elif len(img_list)>len(file_names):
        logger.warning('images more than labels!!!')
        img_wolbl_list = [name for name in img_list if name.split('.')[0] not in lbl_name_wo_suffix]
        arr_img_wolbl = np.array(img_wolbl_list)
        np.savetxt(os.path.join(base_dir, 'dismatch_img_without_lbl.csv'), arr_img_wolbl, fmt='%s')
        dict_stat['图像标注不匹配率'] = f'{round(len(img_wolbl_list)/len(img_list),4)*100}%'



def empty_lbl_check_by_file(lbl_file):
    with open(lbl_file, 'r') as f:
        lines = f.readlines()
    if not lines:
        return True
    else:
        return False


def check_txt(base_dir, dict_stat, mapping_file,  wh_thres=50, split=None, extention='.txt'):
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    
    dict_id_cat = json.load(open(mapping_file))
    # classes = [x for x in dict_id_cat.values()]
    ids = [int(x) for x in dict_id_cat.keys()]
    # dict_cat_id = {classes[ix]:ids[ix] for ix in range(len(classes))}

    for sf in valid_folders:
        if not os.path.isdir(os.path.join(base_dir, sf)):
            continue
        unreadable_files = []
        empty_files = []
        invalid_id_files = []
        uncomplete_files = []
        duplicate_files = []
        invalid_coor_files = []
        suffix_error_files = []
        invalid_wh_ratio_files = []
        lbl_dir = os.path.join(base_dir, sf, 'labels')
        lbl_list = os.listdir(lbl_dir)
        for i_name in lbl_list:
            if not i_name.endswith(extention):
                suffix_error_files.append(i_name)
                logger.error(f'{i_name} extention error!')
                continue

            lbl_file = os.path.join(lbl_dir, i_name)
            try:
                df_lbl = pd.read_csv(lbl_file, delimiter=' ', header=None)
            except:
                logger.error(f'{i_name} is damaged!!')
                unreadable_files.append(i_name)
                continue 

            if empty_lbl_check_by_file(lbl_file):
                logger.warning(f"{i_name} empty label file!")
                empty_files.append(i_name)
                continue

            if df_lbl.shape[1] < 5:
                logger.error(f'{i_name} is uncomplete!')
                uncomplete_files.append(i_name)
                continue

            has_duplicate = df_lbl.duplicated().any()
            if has_duplicate:
                duplicate_files.append(i_name)
                logger.warning(f'{i_name} contains duplicate bbox!')

            
            for ix in range(df_lbl.shape[0]):
                if df_lbl.iloc[ix, 0] not in ids:
                    invalid_id_files.append((i_name, df_lbl.iloc[ix,0]))
                    logger.warning(f"{i_name} contains invalid cat id!")
                    continue

                if df_lbl.iloc[ix, 1:5].max()>=1 or df_lbl.iloc[ix, 1:5].min()<0:
                    logger.warning(f'{i_name} contains invalid coordinates!')
                    invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
                    continue
                elif df_lbl.iloc[ix,3]<=0 or df_lbl.iloc[ix,4]<=0:
                    logger.warning(f'{i_name} contains invalid coordinates!')
                    invalid_coor_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
                    continue
                if max(df_lbl.iloc[ix,3]/max(df_lbl.iloc[ix,4], 0.00001), df_lbl.iloc[ix,4]/max(df_lbl.iloc[ix,3], 0.00001)) > wh_thres:
                    logger.warning(f'{i_name} contains invalid wh ratio')
                    invalid_wh_ratio_files.append((i_name, df_lbl.iloc[ix, 1:5].tolist()))
                    continue

        file_damaged = os.path.join(base_dir, 'all_txts_damaged.json')
        with open(file_damaged, 'w') as f:
            for name in unreadable_files:
                f.write("%s\n" % name)
        f.close() 
        dict_stat['标注文件损坏率'] = f'{round(len(unreadable_files)/len(lbl_list),4)*100}%'     

        file_invalid = os.path.join(base_dir, 'all_txts_invalid_coords.json')
        with open(file_invalid, 'w') as f:
            for name, x1,y1,w1,h1 in invalid_coor_files:
                f.write("%s %.4f %.4f %.4f %.4f\n" % (name, x1,y1,w1,h1))
        f.close() 
        dict_stat['标注不合理比例'] = f'{round(len(invalid_coor_files)/len(lbl_list),4)*100}%'   

        file_err_wh = os.path.join(base_dir, 'all_txts_error_whratio.json')
        with open(file_err_wh, 'w') as f:
            for name, x1,y1,w1,h1 in invalid_wh_ratio_files:
                f.write("%s %.4f %.4f %.4f %.4f\n" % (name, x1,y1,w1,h1))
        f.close()   
        dict_stat['目标边界框宽高比失调率'] = f'{round(len(invalid_wh_ratio_files)/len(lbl_list),4)*100}%'  

        file_empty = os.path.join(base_dir, 'all_txts_empty_lbls.txt')
        with open(file_empty, 'w') as f:
            for name in empty_files:
                f.write("%s\n" % name)
        f.close() 
        dict_stat['空标注比例'] = f'{round(len(empty_files)/len(lbl_list),4)*100}%'  

        file_err_cat = os.path.join(base_dir, 'all_txts_error_cat.txt')
        with open(file_err_cat, 'w') as f:
            for name, cat in invalid_id_files:
                f.write("%s \n" % (name, cat))
        f.close() 
        dict_stat['标注类别错误率'] = f'{round(len(invalid_id_files)/len(lbl_list),4)*100}%'

        with open(os.path.join(base_dir, 'all_txts_lost_key_attributes.txt'), 'w') as f: 
            for name in uncomplete_files:
                f.write("%s\n" % name)
        f.close() 
        dict_stat['标注关键字段缺失率'] = f'{round(len(uncomplete_files)/len(lbl_list),4)*100}%'
        
        with open(os.path.join(base_dir, 'all_txts_extention_error_files.txt'), 'w') as f: 
            for name in suffix_error_files:
                f.write("%s\n" % name)
        f.close() 
        dict_stat['标注文件格式不一致比例'] = f'{round(len(suffix_error_files)/len(lbl_list),4)*100}%'

        with open(os.path.join(base_dir, 'all_txts_duplicate_bbox.json'), 'w') as f: 
            for name in duplicate_files:
                f.write("%s\n" % name)
        f.close() 
        dict_stat['标注重复率'] = f'{round(len(duplicate_files)/len(lbl_list),4)*100}%'