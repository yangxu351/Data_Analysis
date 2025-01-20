import os
from PIL import Image
import numpy as np
import json 
import matplotlib.pyplot as plt
from analysis_scripts.diversity.img_ana import caculate_brightness, calculate_image_contrast
import logging
logger = logging.getLogger(__name__)

plt.get_cmap('Set3')
title_fontdict = {'family' : 'Microsoft YaHei', 'size': 16}
fontdict={"family": "SimHei", "size": 14}


def stat_msk(source_dir, category_imgnum_thresh=20, split=None):
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        src_img_dir = os.path.join(source_dir, sf, 'images')
        src_msk_dir = os.path.join(source_dir, sf, 'masks')    
        ori_imgs = os.listdir(src_img_dir)
        ori_imgs.sort()
        dict_img_suffix = {x.split('.')[0]:x.split('.')[-1] for x in ori_imgs}
        img_num = len(ori_imgs)
        ori_masks = os.listdir(src_msk_dir)
        ori_masks.sort()
        dict_msk_suffix = {x.split('.')[0]:x.split('.')[-1] for x in ori_masks}
        file_names = [x for x in dict_img_suffix.keys() if x in dict_msk_suffix.keys()]
        im_hw_list = []
        dict_cat_imgname = {}
        dict_cat_area = {}

        brightness_avg_list = []
        contrast_list = []

        for ix, im_name in enumerate(file_names):
            pil_img = Image.open(os.path.join(src_img_dir, f'{im_name}.{dict_img_suffix[im_name]}'))
            try:
                brightness = caculate_brightness(pil_img)
            except Exception as e:
                logger.error(f"Unsupported image mode: {str(e)}", exc_info=True)
            brightness_avg_list.append(brightness)

            constrast = calculate_image_contrast(pil_img)
            contrast_list.append(constrast)

            img = np.array(pil_img)
            h, w, _ = img.shape
            im_hw_list.append((h,w))
            # print(os.path.join(src_msk_dir, ori_masks[ix]))
            msk = np.array(Image.open(os.path.join(src_msk_dir, f'{im_name}.{dict_msk_suffix[im_name]}')))
            cat_ids = np.unique(msk)
            for cid in cat_ids:
                cid = int(cid)
                msk_area = np.count_nonzero(msk==cid)
                if cid not in dict_cat_imgname.keys():
                    dict_cat_imgname[cid] = [im_name]
                    dict_cat_area[cid] = [msk_area]
                else:
                    dict_cat_imgname[cid].append(im_name)
                    dict_cat_area[cid].append(msk_area)

        brightness_std = np.std(brightness_avg_list)
        contrast_std = np.std(contrast_list)

        dict_brightness = {'brightness_list': brightness_avg_list, 'brightness_std': brightness_std} 
        with open(os.path.join(source_dir, f'{sf}_allimg_volatility_brightness.json'), 'w') as f: # 亮度波动性
            json.dump(dict_brightness, f, indent=3)

        dict_contrast = {'constrast_list': contrast_list, 'contrast_std':contrast_std} 
        with open(os.path.join(source_dir, f'{sf}_allimg_volatility_contrast.json'), 'w') as f: # 对比度波动性
            json.dump(dict_contrast, f, indent=3)
                    
        arr_img_hw = np.ones((len(im_hw_list), 2), dtype=np.int32)
        for lx, (h, w) in enumerate(im_hw_list):
            arr_img_hw[lx, 0] = h
            arr_img_hw[lx, 1] = w
        np.savetxt(os.path.join(source_dir, f'{sf}_img_hw.csv'), arr_img_hw, delimiter=',', fmt='%.2f') # 图像尺寸多样性

        dict_cat_imgnum = {k:len(v) for k,v in dict_cat_imgname.items()} 
        with open(os.path.join(source_dir, f'{sf}_cat_imgnum.json'), 'w') as f: # 类别多样性
            json.dump(dict_cat_imgnum, f, ensure_ascii=False, indent=3)
        f.close()  

        with open(os.path.join(source_dir, f'{sf}_cat_area.json'), 'w') as f: # 类别面积多样性
            json.dump(dict_cat_area, f, ensure_ascii=False, indent=3)
        f.close()      

        dict_rel_lbl_diver = {k:1.*len(v)/img_num for k,v in dict_cat_imgname.items()}
        with open(os.path.join(source_dir, f'{sf}_relative_cat_diversity.json'), 'w') as f: # 相对类别多样性
            json.dump(dict_rel_lbl_diver, f, ensure_ascii=False, indent=3)
        f.close() 

        dict_lbl_size_diver = {k: 1 if len(v)<category_imgnum_thresh else 0 for k,v in dict_cat_imgname.items()}
        lbl_size_diver = 1.*sum(dict_lbl_size_diver.values())/len(dict_lbl_size_diver.keys())
        with open(os.path.join(source_dir, f'{sf}_cat_size_diversity.txt'), 'w') as f: # 类别大小多样性
            f.write(f'{lbl_size_diver*100}%\n')
        f.close() 


def ana_msk(source_dir, split=None):
    if split:
        valid_folders = [split]
    else:
        valid_folders = ['train', 'val', 'test']
    for sf in valid_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        dict_imgnum = json.load(open(os.path.join(source_dir, f'{sf}_cat_imgnum.json')))
        plt.figure()
        plt.bar(x=dict_imgnum.keys(), height=dict_imgnum.values())
        plt.xlabel('类别', fontdict=fontdict)
        plt.ylabel('图片数量', fontdict=fontdict)
        plt.title('类别多样性', fontdict=title_fontdict)
        plt.savefig(os.path.join(source_dir, f'{sf}_cat_imgnum.jpg'))
        plt.show()

        arr_imghw = np.loadtxt(os.path.join(source_dir, f'{sf}_img_hw.csv'), delimiter=',', dtype=np.int32)
        plt.figure()
        for h,w in arr_imghw:
            plt.scatter(x=w, y=h,  marker='o')
        plt.xlabel('图片宽度', fontdict=fontdict)
        plt.ylabel('图片高度', fontdict=fontdict)
        plt.title('图片尺寸多样性', fontdict=title_fontdict)
        plt.savefig(os.path.join(source_dir, f'{sf}_img_hw.jpg'))
        plt.show()

        dict_cat_area = json.load(open(os.path.join(source_dir, f'{sf}_cat_area.json')))
        plt.figure(figsize=(15,10))
        # colors = np.array(plt.cm.Set2.colors)
        for k,area_list in dict_cat_area.items():
            # for area in area_list:
            #     plt.scatter(x=k, y=area, marker='o')

            ## mean area size
            plt.scatter(x=k, y=np.mean(area_list), marker='o')

        plt.xlabel('类别', fontdict=fontdict, rotation=90)
        plt.ylabel('类别平均面积', fontdict=fontdict)
        plt.title('类别平均面积多样性', fontdict=title_fontdict)
        plt.savefig(os.path.join(source_dir, f'{sf}_cat_area.jpg'))
        plt.show()