import os
import shutil 
import glob 
import numpy as np
import json
import cv2 
from PIL import Image

# https://github.com/yangxu351/xview-Faster-RCNN-with-torchvision/blob/master/data_preprocess/coco_utils.py

def make_folder_if_not(dst_dir, rm_exist=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        if rm_exist:
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)

def rle_decode(counts, mask, cid):
    # print('mask', mask.shape)
    img = np.zeros(mask.shape[0]*mask.shape[1], dtype=np.uint8)
    index = 0
    for ix, count in enumerate(counts):
        if ix%2 !=0: # 奇数位--1
            img[index:index + count] = cid
        # else: # 偶数位--0
        #     img[index:index + count] = 0
        index += count
    return img.reshape(mask.shape[1],mask.shape[0]).T


def convert_coco_to_png(base_dir, src_split='val2017', split='val'):
    '''
        covert coco to png
        find duplicate img
        find duplicate cat
    '''
    js_file = os.path.join(base_dir, 'annotations', f'instances_{src_split}.json')
    js = json.load(open(js_file))
    img_list = js['images']
    dict_img_wh = {}
    dict_img_id_name = {}
    duplicate_imginfo_list = []
    for img_info in img_list:
        # FIXME: check duplicate?
        img_id = img_info['id']
        if img_id in dict_img_id_name.keys():
            print(f'label info of image {img_id} is duplicate!')
            duplicate_imginfo_list.append(img_id)
            continue # duplicate is ignored

        dict_img_wh[img_info['id']] = (img_info['width'], img_info['height'])
        dict_img_id_name[img_info['id']] = img_info['file_name']
    
    dumplicat_imginfo_file = os.path.join(base_dir, f'{split}_duplicate_imginfo.txt')
    with open(dumplicat_imginfo_file, 'w') as f:
        for imgid in duplicate_imginfo_list:
            f.write("%d\n" % (imgid))
    f.close()

    with open(os.path.join(base_dir, 'dict_img_id_name.json'), 'w') as f: # dict of img_id:img_name
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
            print(f'label info of category {cat_id} is duplicate!')
            duplicate_catinfo_list.append(cat_id)
            continue # duplicate is ignored

        dict_cat_id_name[cat_id] = cat_name
        super_cat = cat_info['supercategory']
        if super_cat not in dict_cats_super_children.keys():
            dict_cats_super_children[super_cat] = [cat_name]
        else:
            dict_cats_super_children[super_cat].append(cat_name)

    dumplicat_catinfo_file = os.path.join(base_dir, f'{split}_duplicate_catinfo.txt')
    with open(dumplicat_catinfo_file, 'w') as f:
        for catid in duplicate_catinfo_list:
            f.write("%d\n" % (catid))
    f.close()

    with open(os.path.join(base_dir, f'{split}_dict_cat_super_children.json'), 'w') as f: # dict of supercategory:[cat1,cat2,...]
        json.dump(dict_cats_super_children, f, ensure_ascii=False, indent=3)
    f.close() 

    with open(os.path.join(base_dir, f'{split}_dict_cat_id_name.json'), 'w') as f: # dict of cat_id:cat_name
        json.dump(dict_cat_id_name, f, ensure_ascii=False, indent=3)
    f.close() 

    anns_list = js['annotations']
    dict_imgname_segs = {}
    for ann in anns_list:
        img_id = ann['image_id']
        img_name = dict_img_id_name[img_id]
        cat_id = ann['category_id']
        iscrowd = ann['iscrowd']
        segmentations = ann['segmentation']
        if img_name not in dict_imgname_segs.keys():
            dict_imgname_segs[img_name] = [(cat_id, iscrowd, segmentations)]
        else:
            dict_imgname_segs[img_name].append((cat_id, iscrowd, segmentations))
    
    mask_dir = os.path.join(base_dir, split, 'masks')
    make_folder_if_not(mask_dir)
    image_dir = os.path.join(base_dir, split,'images')
    img_list = os.listdir(image_dir)
    for img_name, cid_crowd_segs in dict_imgname_segs.items():
        # for test
        if img_name not in img_list:
            continue
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.int32)
        rle_msk = None
        for cid, iscrowd, segs in cid_crowd_segs:
            if iscrowd: # 1 RLE
                seg_rle = segs['counts']
                rle_msk = rle_decode(seg_rle, mask, cid)
                # print('rle msk', rle_msk.shape)
            else: # 0 Poly
                for seg in segs:
                    # getting the points
                    xs = seg[0::2]
                    ys = seg[1::2]
                    pts = np.array([[x, y] for x, y in zip(xs, ys)], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # draw the points on the mask image.
                    try:
                        mask = cv2.fillPoly(mask, [pts], cid)
                    except Exception as e:
                        print(f"[ERROR] error for {img_name}, len={len(pts)}")
            if not rle_msk is None:
                mask[rle_msk!=0] = rle_msk[rle_msk!=0]
            cv2.imwrite(os.path.join(mask_dir, img_name), mask)


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
    convert_coco_to_png(base_dir, src_split='val2017', split='val') 
    move_imgs_to_specified_folder(base_dir, src_split='val2017', split='val')