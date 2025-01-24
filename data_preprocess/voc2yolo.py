import os
import shutil 
import glob 
import pandas as pd
import numpy as np
import json 
import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)


def make_folder_if_not(dst_dir, rm_exist=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        if rm_exist:
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)

def dose_match_img_lbl(base_dir, dict_stat):
    xml_dir = os.path.join(base_dir, 'Annotations')
    img_dir = os.path.join(base_dir, 'JPEGImages')
    
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_name_wo_suffix = [f.split('.')[0] for f in img_list]
    img_wolbl_list = None 

    lbl_list = os.listdir(xml_dir)
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


def covert_voc_to_yolo(base_dir, dict_stat, mapping_file, wh_thres=50, extention='.xml'):
    ''' 
        convert VOC to YOLO 
        empty labels
        duplicate annotations
    '''
    xml_dir = os.path.join(base_dir, 'Annotations')
    txt_dir = os.path.join(base_dir, 'labelsAll') 
    make_folder_if_not(txt_dir)

    # xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    xml_files = os.listdir(xml_dir)
    
    dict_id_cat = json.load(open(mapping_file))
    classes = [x for x in dict_id_cat.values()]
    ids = [int(x) for x in dict_id_cat.keys()]
    dict_cat_id = {classes[ix]:ids[ix] for ix in range(len(classes))}

    index = 0
    dict_duplicate_file_box = {}
    empty_lbls = []
    dict_invalid_bbx_xmls = {}
    invalid_cat_files = []
    error_wh_ratio_xmls = []
    attribute_lost_xmls = []
    suffix_error_files = []
    damaged_xmls = []
    for file_name in xml_files:
        if not file_name.endswith(extention):
            suffix_error_files.append(file_name)
            logger.error(f'{file_name} extention error!')
            continue
        xf = os.path.join(xml_dir, file_name)
        try:
            ann_tree = ET.parse(xf)
            ann_root = ann_tree.getroot()
        except:
            logger.error('xml file is not readable')
            damaged_xmls.append(file_name)
            continue 

        try:
            size = ann_root.find('size')
            img_width = int(size.findtext('width'))
            img_height = int(size.findtext('height'))
        except:
            logger.error('key attributes of size, width, height lost')
            # logger.error(str(e), exc_info=True)
            attribute_lost_xmls.append(file_name)
            continue
            
        objects = ann_root.findall('object')
        lbl_list = []
        oribox_list = []
        if not len(objects):
            empty_lbls.append(file_name)
        for obj in objects:
            cat = obj.findtext('name')
            if cat not in classes:
                invalid_cat_files.append(file_name)
                logger.error(f'{file_name} contains invalid catid!')

            bndbox = obj.find('bndbox')
            # check invlid coordinates
            if bndbox is None: # FIXME:  box None 
                if file_name not in dict_invalid_bbx_xmls.keys():
                    dict_invalid_bbx_xmls[file_name]= [[cat, None, None, None, None]]
                else:
                    dict_invalid_bbx_xmls[file_name].append([cat, None, None, None, None])
                continue
            sxmin = bndbox.findtext('xmin')
            symin = bndbox.findtext('ymin')
            sxmax = bndbox.findtext('xmax')
            symax = bndbox.findtext('ymax')
            
            xmin = float(sxmin) - 1 if sxmin else None
            ymin = float(symin) - 1 if symin else None
            xmax = float(sxmax) if sxmax else None
            ymax = float(symax) if symax else None
           
            # check invlid coordinates
            if xmin is None or ymin is None or xmax is None or ymax is None: # FIXME: None 
                if file_name not in dict_invalid_bbx_xmls.keys():
                    dict_invalid_bbx_xmls[file_name]= [[cat,xmin, ymin, xmax, ymax]]
                else:
                    dict_invalid_bbx_xmls[file_name].append([cat,xmin, ymin, xmax, ymax])
                logger.error(f'{file_name} contains invalid coordinates!')
                continue
            
            box = [xmin, ymin, xmax, ymax]
            width = xmax - xmin 
            height = ymax - ymin
            # check invlid coordinates
            if xmin < 0 or ymin < 0 or xmax >img_width or ymax >img_height or width<=0 or height<=0:
                if file_name not in dict_invalid_bbx_xmls.keys():
                    dict_invalid_bbx_xmls[file_name]= [[cat,xmin, ymin, xmax, ymax]]
                else:
                    dict_invalid_bbx_xmls[file_name].append([cat,xmin, ymin, xmax, ymax])
                logger.error(f'{file_name} contains invalid coordinates!')
                continue
            elif max(width/height, height/width) > wh_thres:
                error_wh_ratio_xmls.append(file_name)
                logger.error(f'{file_name} contains invalid wh ratio!')
                continue
           
            # check duplicate
            if box not in oribox_list:
                oribox_list.append(box)
            else:
                if file_name not in dict_duplicate_file_box.keys():
                    dict_duplicate_file_box[file_name] = np.array(box, dtype=np.int32).tolist()
                else:
                    dict_duplicate_file_box[file_name].append(np.array(box, dtype=np.int32).tolist())

            center_wid = xmin+width/2.
            center_hei = ymin+height/2.
            lbl_list.append([dict_cat_id[cat], center_wid/img_width, center_hei/img_height, width/img_width, height/img_height])
        
        # print('index', index)
        
        lbl_file = os.path.join(txt_dir, os.path.basename(xf).replace(extention, '.txt'))
        with open(lbl_file, 'w') as f:
            for cat_id, cx,cy,w,h in lbl_list:
                f.write("%d %.4f %.4f %.4f %.4f\n" % (cat_id, cx, cy, w, h))
        f.close() 


    file_damaged_xml = os.path.join(base_dir, 'all_xmls_damaged.json')
    with open(file_damaged_xml, 'w') as f:
        for name in damaged_xmls:
            f.write("%s\n" % name)
    f.close() 
    dict_stat['标注文件损坏率'] = f'{round(len(damaged_xmls)/len(xml_files),4)*100}%'

    file_invalid_xml = os.path.join(base_dir, 'all_xmls_invalid_coords.json')
    with open(file_invalid_xml, 'w') as f:
        json.dump(dict_invalid_bbx_xmls, f, ensure_ascii=False, indent=3)
    f.close()   
    dict_stat['标注不合理比例'] = f'{round(len(dict_invalid_bbx_xmls.keys())/len(xml_files),4)*100}%'
    
    file_err_wh_xml = os.path.join(base_dir, 'all_xmls_error_whratio.json')
    with open(file_err_wh_xml, 'w') as f:
        for name in error_wh_ratio_xmls:
            f.write("%s\n" % name)
    f.close()   
    dict_stat['目标边界框宽高比失调率'] = f'{round(len(error_wh_ratio_xmls)/len(xml_files),4)*100}%'

    file_empty = os.path.join(base_dir, 'all_xmls_empty_lbls.txt')
    with open(file_empty, 'w') as f:
        for name in empty_lbls:
            f.write("%s\n" % name)
    f.close() 
    dict_stat['空标注比例'] = f'{round(len(empty_lbls)/len(xml_files),4)*100}%'

    file_err_cat = os.path.join(base_dir, 'all_xmls_error_cat.txt')
    with open(file_err_cat, 'w') as f:
        for name in invalid_cat_files:
            f.write("%s\n" % name)
    f.close() 
    dict_stat['标注类别错误率'] = f'{round(len(invalid_cat_files)/len(xml_files),4)*100}%'
 
    with open(os.path.join(base_dir, 'all_xmls_lost_key_attributes.txt'), 'w') as f: 
        for name in attribute_lost_xmls:
            f.write("%s\n" % name)
    f.close() 
    dict_stat['标注关键字段缺失率'] = f'{round(len(attribute_lost_xmls)/len(xml_files),4)*100}%'
    
    with open(os.path.join(base_dir, 'all_xmls_extention_error_files.txt'), 'w') as f: 
        for name in suffix_error_files:
            f.write("%s\n" % name)
    f.close() 
    dict_stat['标注文件格式不一致比例'] = f'{round(len(suffix_error_files)/len(xml_files),4)*100}%'

    with open(os.path.join(base_dir, 'all_xmls_duplicate_bbox.json'), 'w') as f: # dict of file_name:[bbox]
        json.dump(dict_duplicate_file_box, f, ensure_ascii=False, indent=3)
    f.close() 
    dict_stat['标注重复率'] = f'{round(len(dict_duplicate_file_box.keys())/len(xml_files),4)*100}%'

def movefiles_by_setfile(base_dir, split='train'):
    '''
    create new folder and move files to specified folder according to the set file
    '''
    set_dir = os.path.join(base_dir, 'ImageSets/Main')
    img_dir = os.path.join(base_dir, 'JPEGImages')
    txt_dir = os.path.join(base_dir, 'labelsAll') 

    set_file = os.path.join(set_dir, f'{split}.txt')
    df_set = pd.read_csv(set_file, header=None)
    img_foler = os.path.join(base_dir, split, 'images')
    make_folder_if_not(img_foler)
    lbl_foler = os.path.join(base_dir, split, 'labels')
    make_folder_if_not(lbl_foler)
    for name in df_set.iloc[:,0]:
        img_name = f'{name}.jpg'
        lbl_name = f'{name}.txt'
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(img_foler, img_name))
        shutil.copy(os.path.join(txt_dir, lbl_name), os.path.join(lbl_foler, lbl_name))

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/BCCD_VOC'

    
    covert_voc_to_yolo(base_dir)

    
    movefiles_by_setfile(base_dir, split='train')
    movefiles_by_setfile(base_dir, split='val')
    movefiles_by_setfile(base_dir, split='test')
    
    
        
