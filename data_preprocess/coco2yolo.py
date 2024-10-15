import os
import shutil 
import json


def make_folder_if_not(dst_dir, rm_exist=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        if rm_exist:
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)


def convert_coco_to_yolo(base_dir, src_split='val2017', split='val'):
    js_file = os.path.join(base_dir, 'annotations', f'instances_{src_split}.json')
    js = json.load(open(js_file))
    img_list = js['images']
    dict_img_wh = {}
    dict_img_id_name = {}
    for img_info in img_list:
        dict_img_wh[img_info['id']] = (img_info['width'], img_info['height'])
        dict_img_id_name[img_info['id']] = img_info['file_name']

    with open(os.path.join(base_dir, f'{split}_dict_img_id_name.json'), 'w') as f: # dict of img_id:img_name
        json.dump(dict_img_id_name, f, ensure_ascii=False, indent=3)
    f.close() 

    categories = js['categories']
    dict_cats_super_children = {}
    dict_cat_id_name = {}
    for cat_info in categories:
        cat_id = cat_info['id']
        cat_name = cat_info['name']
        dict_cat_id_name[cat_id] = cat_name
        super_cat = cat_info['supercategory']
        if super_cat not in dict_cats_super_children.keys():
            dict_cats_super_children[super_cat] = [cat_name]
        else:
            dict_cats_super_children[super_cat].append(cat_name)
    
    with open(os.path.join(base_dir, f'{split}_dict_cat_super_children.json'), 'w') as f: # dict of supercategory:[cat1,cat2,...]
        json.dump(dict_cats_super_children, f, ensure_ascii=False, indent=3)
    f.close() 

    with open(os.path.join(base_dir, f'{split}_dict_cat_id_name.json'), 'w') as f: # dict of cat_id:cat_name
        json.dump(dict_cat_id_name, f, ensure_ascii=False, indent=3)
    f.close() 

    anns_list = js['annotations']
    dict_imgname_anns = {}
    for ann in anns_list:
        img_id = ann['image_id']
        img_name = dict_img_id_name[img_id]
        cat_id = ann['category_id']
        bbox = ann['bbox'] # tl_x tl_y w h
        img_width, img_height = dict_img_wh.get(img_id)
        cx = (bbox[0] + bbox[2]/2. - 1)/img_width
        cy = (bbox[1] + bbox[3]/2. - 1)/img_height
        reg_w = bbox[2]/img_width
        reg_h = bbox[3]/img_height
        if img_name not in dict_imgname_anns.keys():
            dict_imgname_anns[img_name] = [[cat_id, cx, cy, reg_w, reg_h]]
        else:
            dict_imgname_anns[img_name].append([cat_id, cx, cy, reg_w, reg_h])
        
    dst_lbl_dir = os.path.join(base_dir, split, 'labels')
    make_folder_if_not(dst_lbl_dir)
    for img_name, anns in dict_imgname_anns.items():
        lbl_name = img_name.replace('.jpg', '.txt')
        lbl_file = os.path.join(dst_lbl_dir, lbl_name)
        with open(lbl_file, 'w') as f:
            for cid, cx, cy, w, h in anns:
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (cid, cx, cy, w, h))
        f.close()

def move_imgs_to_specified_folder(base_dir, src_split='val2017', split='val'):    
    '''
        adjust the organization of original data to specified folder
    '''
    dst_img_dir = os.path.join(base_dir, split, 'images')
    make_folder_if_not(dst_img_dir)
    src_img_dir = os.path.join(base_dir, 'images', src_split)
    src_ime_names = os.listdir(src_img_dir)
    for f_name in src_ime_names:
        shutil.copy(os.path.join(src_img_dir, f_name), os.path.join(dst_img_dir, f_name))

if __name__ == '__main__':
    base_dir = 'F:/Public_Dataset/COCO'
    
    convert_coco_to_yolo(base_dir, src_split='val2017', split='val')
    # move_imgs_to_specified_folder(base_dir, src_split='val2017', split='val')