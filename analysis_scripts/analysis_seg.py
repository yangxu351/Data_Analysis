import os
from PIL import Image
import numpy as np
import json 
import matplotlib.pyplot as plt
plt.get_cmap('Set3')
title_fontdict = {'family' : 'Microsoft YaHei', 'size': 16}
fontdict={"family": "SimHei", "size": 14}


def stat_msk(source_dir, label_imgnum_thresh=20):
    split_folders = ['train', 'val', 'test']
    for sf in split_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        src_img_dir = os.path.join(source_dir, sf, 'images')
        src_msk_dir = os.path.join(source_dir, sf, 'masks')    
        ori_names = os.listdir(src_img_dir)
        ori_names.sort()
        img_num = len(ori_names)
        ori_masks = os.listdir(src_msk_dir)
        ori_masks.sort()
        im_hw_list = []
        dict_cat_imgname = {}
        dict_cat_area = {}
        for ix, im_name in enumerate(ori_names):
            img = np.array(Image.open(os.path.join(src_img_dir, im_name)))
            h, w, _ = img.shape
            im_hw_list.append((h,w))
            msk = np.array(Image.open(os.path.join(src_msk_dir, ori_masks[ix])))
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
                
        arr_img_hw = np.ones((len(im_hw_list), 2))
        for lx, (h, w) in enumerate(im_hw_list):
            arr_img_hw[lx, 0] = h
            arr_img_hw[lx, 1] = w
        np.savetxt(os.path.join(source_dir, f'{sf}_img_hw.csv'), arr_img_hw) # 图像尺寸多样性

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

        dict_lbl_size_diver = {k: 1 if len(v)<label_imgnum_thresh else 0 for k,v in dict_cat_imgname.items()}
        lbl_size_diver = 1.*sum(dict_lbl_size_diver.values())/len(dict_lbl_size_diver.keys())
        with open(os.path.join(source_dir, f'{sf}_cat_size_diversity.txt'), 'w') as f: # 类别大小多样性
            f.write(f'{lbl_size_diver*100}%\n')
        f.close() 


def ana_msk(source_dir):
    split_folders = os.listdir(source_dir)
    for sf in split_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        if sf in ['train', 'val', 'test']:
            dict_imgnum = json.load(open(os.path.join(source_dir, f'{sf}_cat_imgnum.json')))
            plt.figure()
            plt.bar(x=dict_imgnum.keys(), height=dict_imgnum.values())
            plt.xlabel('类别', fontdict=fontdict)
            plt.ylabel('图片数量', fontdict=fontdict)
            plt.title('类别多样性', fontdict=title_fontdict)
            plt.savefig(os.path.join(source_dir, f'{sf}_cat_imgnum.jpg'))
            plt.show()

            arr_imghw = np.loadtxt(os.path.join(source_dir, f'{sf}_img_hw.csv'))
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
            colors = plt.cm.Set2.colors
            for k,area_list in dict_cat_area.items():
                for area in area_list:
                    plt.scatter(x=k, y=area, marker='o', c=colors[int(k)])
            plt.xlabel('类别', fontdict=fontdict)
            plt.ylabel('类别面积', fontdict=fontdict)
            plt.title('类别面积多样性', fontdict=title_fontdict)
            plt.savefig(os.path.join(source_dir, f'{sf}_cat_area.jpg'))
            plt.show()