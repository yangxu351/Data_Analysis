import os
import shutil 
import glob 
import pandas as pd
import xml.etree.ElementTree as ET

def make_folder_if_not(dst_dir, rm_exist=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        if rm_exist:
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)


def covert_voc_to_yolo(base_dir):
    ''' convert VOC to YOLO '''
    xml_dir = os.path.join(base_dir, 'Annotations')
    txt_dir = os.path.join(base_dir, 'labelsAll') 

    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    dict_cats = {}
    index = 0
    make_folder_if_not(txt_dir)
    for xf in xml_files:
        ann_tree = ET.parse(xf)
        ann_root = ann_tree.getroot()
        size = ann_root.find('size')
        img_width = int(size.findtext('width'))
        img_height = int(size.findtext('height'))
        objects = ann_root.findall('object')
        lbl_list = []
        for obj in objects:
            cat = obj.find('name')
            if cat not in dict_cats.keys():
                dict_cats[cat] = index
                index += 1
            bndbox = obj.find('bndbox')
            xmin = max(float(bndbox.findtext('xmin')) - 1, 0.)
            ymin = max(float(bndbox.findtext('ymin')) - 1, 0.)
            xmax = float(bndbox.findtext('xmax'))
            ymax = float(bndbox.findtext('ymax'))
            width = xmax - xmin 
            height = ymax - ymin 
            center_wid = xmin+width/2.
            center_hei = ymin+height/2.
            lbl_list.append([dict_cats.get(cat), center_wid/img_width, center_hei/img_height, width/img_width, height/img_height])
        
        lbl_file = os.path.join(txt_dir, os.path.basename(xf).replace('.xml', '.txt'))
        with open(lbl_file, 'w') as f:
            for id, cx,cy,w,h in lbl_list:
                f.write("%d\t%.4f\t%.4f\t%.4f\t%.4f\n" % (id, cx, cy, w, h))
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
    
    
        
