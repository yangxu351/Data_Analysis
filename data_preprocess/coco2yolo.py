import os
import sys
import shutil 
import json
import logging
logger = logging.getLogger(__name__)


def make_folder_if_not(dst_dir, rm_exist=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        if rm_exist:
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)


def convert_coco_to_yolo(base_dir, src_split='val2017', split='val'):
    '''
    conver coco format to yolo
    find duplicate cat info
    find duplicate image info 
    find duplicate annoation
    '''
    js_file = os.path.join(base_dir, 'annotations', f'instances_{src_split}.json')
    try:
        js = json.load(open(js_file))
        img_list = js['images']
    except Exception as e:
         logger.error(f"{str(e)}", exc_info=True)
         sys.exit(1)
         
    dict_img_wh = {}
    dict_img_id_name = {}
    
    duplicate_imginfo_list = []
    attribute_lost_imgid_list = []
    
    for img_info in img_list:
        img_id = img_info['id']
        # FIXME: check duplicate?
        if img_id in dict_img_id_name.keys():
            logger.warning(f'label info of image {img_id} is duplicate!')
            duplicate_imginfo_list.append(img_id)
            continue # duplicate is ignored

        if 'file_name' not in img_info.keys() or 'width' not in img_info.keys() or 'height' not in img_info.keys():
            logger.error('key attributes of images are not found') 
            attribute_lost_imgid_list.append(img_id)
            continue
        dict_img_id_name[img_id] = img_info['file_name']
        dict_img_wh[img_id] = (img_info['width'], img_info['height'])
            
    attribute_lost_img_file = os.path.join(base_dir, split, 'attribute_lost_imgid.txt') # 关键字段缺失
    with open(attribute_lost_img_file, 'w') as f:
        for imgid in attribute_lost_imgid_list:
            f.write("%d\n" % (imgid))
    f.close()

    dumplicat_imginfo_file = os.path.join(base_dir, split, 'duplicate_imginfo.txt') # ID重复
    with open(dumplicat_imginfo_file, 'w') as f:
        for imgid in duplicate_imginfo_list:
            f.write("%d\n" % (imgid))
    f.close()

    with open(os.path.join(base_dir, f'{split}_dict_img_id_name.json'), 'w') as f: # dict of img_id:img_name
        json.dump(dict_img_id_name, f, ensure_ascii=False, indent=3)
    f.close() 

    categories = js['categories']
    dict_cats_super_children = {}
    dict_cat_id_name = {}
    duplicate_catinfo_list = []
    for cat_info in categories:
        cat_id = cat_info['id']
        cat_name = cat_info['name']
        # FIXME: check duplicate?
        if cat_id in dict_cat_id_name.keys():
            logger.warning(f'label info of category {cat_id} is duplicate!')
            duplicate_catinfo_list.append(cat_info)
            continue # duplicate is ignored

        dict_cat_id_name[cat_id] = cat_name
        if 'supercategory' in cat_info.keys():
            super_cat = cat_info['supercategory']
            if super_cat not in dict_cats_super_children.keys():
                dict_cats_super_children[super_cat] = [cat_name]
            else:
                dict_cats_super_children[super_cat].append(cat_name)

    dumplicat_catinfo_file = os.path.join(base_dir, f'{split}_duplicate_catinfo.txt') # 父类别重复
    with open(dumplicat_catinfo_file, 'w') as f:
        for catinfo in duplicate_catinfo_list:
            f.write("%s\n" % (catinfo))
    f.close()

    with open(os.path.join(base_dir, f'{split}_dict_cat_super_children.json'), 'w') as f: # dict of supercategory:[cat1,cat2,...]
        json.dump(dict_cats_super_children, f, ensure_ascii=False, indent=3)
    f.close() 

    with open(os.path.join(base_dir, f'{split}_dict_cat_id_name.json'), 'w') as f: # dict of cat_id:cat_name
        json.dump(dict_cat_id_name, f, ensure_ascii=False, indent=3)
    f.close() 

    anns_list = js['annotations']
    dict_imgname_anns = {}
    
    duplicate_imgid_annid_bbox = []
    dict_imgid_bbox = {}
    dict_invalid_bbox = {}
    for ann in anns_list:
        ann_id = ann['id']
        img_id = ann['image_id']
        img_name = dict_img_id_name[img_id]
        cat_id = ann['category_id']
        # check bbox
        if 'bbox' not in ann.keys():
            dict_invalid_bbox[ann_id] = None
            continue
        else:
            bbox = ann['bbox'] # tl_x tl_y w h
            if len(bbox) !=4:
                dict_invalid_bbox[ann_id] = bbox
                continue

        if img_id not in dict_imgid_bbox.keys():
            dict_imgid_bbox[img_id] = [bbox]
        else: # duplicate bbox
            if bbox in dict_imgid_bbox[img_id]:
                dict_imgid_bbox[img_id].append(bbox)
                duplicate_imgid_annid_bbox.append((img_id, ann_id, bbox))

        img_width, img_height = dict_img_wh.get(img_id)

        # check bbox
        if any([x<0 for x in bbox]):
            dict_invalid_bbox[ann_id] = bbox
        if bbox[0]+bbox[2] > img_width or bbox[1]+bbox[3] > img_height:
            dict_invalid_bbox[ann_id] = bbox

        cx = (bbox[0] + bbox[2]/2. - 1)/img_width
        cy = (bbox[1] + bbox[3]/2. - 1)/img_height
        reg_w = bbox[2]/img_width
        reg_h = bbox[3]/img_height
        cid_cxy_wh = [cat_id, cx, cy, reg_w, reg_h]
        if img_name not in dict_imgname_anns.keys():
            dict_imgname_anns[img_name] = [cid_cxy_wh]
        else:
            dict_imgname_anns[img_name].append(cid_cxy_wh)

    with open(os.path.join(base_dir, f'{split}_ori_invalid_annid_bbox.json'), 'w') as f:
        json.dump(dict_invalid_bbox, f, ensure_ascii=False, indent=3)
        f.close()
    
    dumplicate_anno_file = os.path.join(base_dir, f'{split}_ori_duplicate_imgid_annid_bbox.txt')
    with open(dumplicate_anno_file, 'w') as f:
        for ix, (imgid, annid, box) in enumerate(duplicate_imgid_annid_bbox):
            if ix==0:
                f.write("%s,%s,%s,%s,%s,%s\n" % ("image_id", "ann_id", "bbx_tlx", "bbx_tly", "bbx_w", "bbx_h"))
                f.write("%d,%d,%.4f,%.4f,%.4f,%.4f\n" % (imgid, annid, box[0], box[1], box[2], box[3]))
            else:
                f.write("%d,%d,%.4f,%.4f,%.4f,%.4f\n" % (imgid, annid, box[0], box[1], box[2], box[3]))
    f.close()

    dst_lbl_dir = os.path.join(base_dir, split, 'labels')
    make_folder_if_not(dst_lbl_dir)
    for img_name, anns in dict_imgname_anns.items():
        lbl_name = img_name.replace('.jpg', '.txt')
        lbl_file = os.path.join(dst_lbl_dir, lbl_name)
        with open(lbl_file, 'w') as f:
            for cid, cx, cy, w, h in anns:
                f.write('%d,%.4f,%.4f,%.4f,%.4f\n' % (cid, cx, cy, w, h))
        f.close()

def move_imgs_to_specified_folder(base_dir, src_split='val2017', split='val'):    
    '''
        adjust the organization of original data to specified folder
    '''
    dst_img_dir = os.path.join(base_dir, split, 'images')
    make_folder_if_not(dst_img_dir)
    src_img_dir = os.path.join(base_dir, src_split, 'images')
    src_ime_names = os.listdir(src_img_dir)
    for f_name in src_ime_names:
        shutil.copy(os.path.join(src_img_dir, f_name), os.path.join(dst_img_dir, f_name))

if __name__ == '__main__':
    base_dir = 'F:/Public_Dataset/COCO'
    
    convert_coco_to_yolo(base_dir, src_split='val2017', split='val')
    # move_imgs_to_specified_folder(base_dir, src_split='val2017', split='val')