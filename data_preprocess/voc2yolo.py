import os
import shutil 
import glob 
import pandas as pd
import numpy as np
import json 
import xml.etree.ElementTree as ET

def make_folder_if_not(dst_dir, rm_exist=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        if rm_exist:
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)


def covert_voc_to_yolo(base_dir):
    ''' 
        convert VOC to YOLO 
        empty labels
        duplicate annotations
    '''
    xml_dir = os.path.join(base_dir, 'Annotations')
    txt_dir = os.path.join(base_dir, 'labelsAll') 
    make_folder_if_not(txt_dir)

    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    dict_cats = {}
    index = 0
    dict_duplicate_file_box = {}
    empty_lbls = []
    dict_invalid_bbx_xmls = {}
    
    for xf in xml_files:
        file_name = os.path.basename(xf)
        ann_tree = ET.parse(xf)
        ann_root = ann_tree.getroot()
        size = ann_root.find('size')
        img_width = int(size.findtext('width'))
        img_height = int(size.findtext('height'))
        objects = ann_root.findall('object')
        lbl_list = []
        oribox_list = []
        if not len(objects):
            empty_lbls.append(file_name)
        for obj in objects:
            cat = obj.findtext('name')
            if cat not in dict_cats.keys():
                dict_cats[cat] = index
                index += 1
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
            if xmin is None or ymin is None or xmax is None or ymax is None: # FIXME: None is reaplaced by -10000
                if file_name not in dict_invalid_bbx_xmls.keys():
                    dict_invalid_bbx_xmls[file_name]= [[cat,xmin, ymin, xmax, ymax]]
                else:
                    dict_invalid_bbx_xmls[file_name].append([cat,xmin, ymin, xmax, ymax])
                continue
            
            # check invlid coordinates
            if xmin < 0 or ymin < 0 or xmax >img_width or ymax >img_height:
                if file_name not in dict_invalid_bbx_xmls.keys():
                    dict_invalid_bbx_xmls[file_name]= [[cat,xmin, ymin, xmax, ymax]]
                else:
                    dict_invalid_bbx_xmls[file_name].append([cat,xmin, ymin, xmax, ymax])
                continue

            box = [xmin, ymin, xmax, ymax]
            # check duplicate
            if box not in oribox_list:
                oribox_list.append(box)
            else:
                if file_name not in dict_duplicate_file_box.keys():
                    dict_duplicate_file_box[file_name] = np.array(box, dtype=np.int32).tolist()
                else:
                    dict_duplicate_file_box[file_name].append(np.array(box, dtype=np.int32).tolist())
            
            width = xmax - xmin 
            height = ymax - ymin 
            center_wid = xmin+width/2.
            center_hei = ymin+height/2.
            lbl_list.append([dict_cats[cat], center_wid/img_width, center_hei/img_height, width/img_width, height/img_height])
        
        # print('index', index)
        
        lbl_file = os.path.join(txt_dir, os.path.basename(xf).replace('.xml', '.txt'))
        with open(lbl_file, 'w') as f:
            for cat_id, cx,cy,w,h in lbl_list:
                f.write("%d,%.4f,%.4f,%.4f,%.4f\n" % (cat_id, cx, cy, w, h))
        f.close() 

        file_invalid_xml = os.path.join(base_dir, 'all_files_invalid_coords.json')
        with open(file_invalid_xml, 'w') as f:
            json.dump(dict_invalid_bbx_xmls, f, ensure_ascii=False, indent=3)
        f.close()   

        file_empty = os.path.join(base_dir, 'all_files_without_lbls.txt')
        with open(file_empty, 'w') as f:
            for name in empty_lbls:
                f.write("%s\n" % name)
        f.close()  

        with open(os.path.join(base_dir, 'all_duplicate_file_bbox.json'), 'w') as f: # dict of file_name:[bbox]
            json.dump(dict_duplicate_file_box, f, ensure_ascii=False, indent=3)
        f.close() 

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
    
    
        
