import os
from PIL import Image
import numpy as np
import pandas as pd
import json 
import matplotlib.pyplot as plt
plt.get_cmap('Set3')
title_fontdict = {'family' : 'Microsoft YaHei', 'size': 16}
fontdict={"family": "SimHei", "size": 14}


def get_bbx_wh(lbl, w, h):
    lbl.iloc[:, 1] *= w
    lbl.iloc[:, 2] *= h
    lbl.iloc[:, 3] *= w
    lbl.iloc[:, 4] *= h
    arr_lbl = lbl.to_numpy()
    # return arr_lbl[:,0].astype(np.int16), np.round(arr_lbl[:, 3:5], decimals=2)
    return np.array(list(zip(lbl.iloc[:, 0], np.round(arr_lbl[:, 3], decimals=2), np.round(arr_lbl[:, 4], decimals=2))))

def stat_odt(source_dir, label_imgnum_thresh=20, dst_lbl_suffix='.txt'):
    split_folders = ['train', 'val', 'test']
    for sf in split_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        src_img_dir = os.path.join(source_dir, sf, 'images')
        src_lbl_dir = os.path.join(source_dir, sf, 'labels')    
        ori_imgs = os.listdir(src_img_dir)
        ori_imgs.sort()
        img_num = len(ori_imgs)
        im_hw_list = []
        dict_cat_bbxhw = {}
        dict_cat_imgname = {}
        for ix, im_name in enumerate(ori_imgs):
            img = np.array(Image.open(os.path.join(src_img_dir, im_name)))
            # print(' ori shape',  img.shape) # h,w,c
            h, w, _ = img.shape
            im_hw_list.append((h,w))

            lbl_file = os.path.join(src_lbl_dir, f"{im_name.split('.')[0]}{dst_lbl_suffix}")
            lbl = pd.read_csv(lbl_file, header=None, delimiter=' ')
            arr_cat_bbxwh = get_bbx_wh(lbl, w, h)
            for cx in range(arr_cat_bbxwh.shape[0]):
                cat_id = int(arr_cat_bbxwh[cx, 0])
                if cat_id not in dict_cat_bbxhw.keys():
                    dict_cat_bbxhw[cat_id] = [(arr_cat_bbxwh[cx, 1],arr_cat_bbxwh[cx, 2])]
                    dict_cat_imgname[cat_id] = [im_name]
                else:
                    dict_cat_bbxhw[cat_id].append((arr_cat_bbxwh[cx, 1],arr_cat_bbxwh[cx, 2]))
                    if im_name not in dict_cat_imgname[cat_id]:
                        dict_cat_imgname[cat_id].append(im_name)
        arr_img_hw = np.ones((len(im_hw_list), 2))
        for lx, (h, w) in enumerate(im_hw_list):
            arr_img_hw[lx, 0] = h
            arr_img_hw[lx, 1] = w
        np.savetxt(os.path.join(source_dir, f'{sf}_img_hw.csv'), arr_img_hw) # 图像尺寸多样性

        with open(os.path.join(source_dir, f'{sf}_cat_bbxhw.json'), 'w') as f: # 类别尺寸多样性
            json.dump(dict_cat_bbxhw, f, ensure_ascii=False, indent=3)
        f.close()

        dict_cat_imgnum = {k:len(v) for k,v in dict_cat_imgname.items()}
        with open(os.path.join(source_dir, f'{sf}_cat_imgnum.json'), 'w') as f: # 类别多样性
            json.dump(dict_cat_imgnum, f, ensure_ascii=False, indent=3)
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

def ana_odt(source_dir):
    split_folders = os.listdir(source_dir)
    for sf in split_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        if sf in ['train', 'val', 'test']:
            arr_img_hw = np.loadtxt(os.path.join(source_dir, f'{sf}_img_hw.csv'))
            plt.figure()
            for h,w in arr_img_hw:
                plt.scatter(x=w, y=h,  marker='o')
            plt.xlabel('图片宽度', fontdict=fontdict)
            plt.ylabel('图片高度', fontdict=fontdict)
            plt.title('图片尺寸多样性', fontdict=title_fontdict)
            plt.savefig(os.path.join(source_dir, f'{sf}_img_hw.jpg'))
            plt.show()

            dict_imgnum = json.load(open(os.path.join(source_dir, f'{sf}_cat_imgnum.json')))
            plt.figure()
            plt.bar(x=dict_imgnum.keys(), height=dict_imgnum.values())
            plt.xlabel('类别', fontdict=fontdict)
            plt.xticks(rotation=90)
            plt.ylabel('图片数量', fontdict=fontdict)
            plt.title('类别多样性', fontdict=title_fontdict)
            plt.savefig(os.path.join(source_dir, f'{sf}_cat_imgnum.jpg'))
            plt.show()

            dict_cat_bbxhw = json.load(open(os.path.join(source_dir, f'{sf}_cat_bbxhw.json')))
            plt.figure(figsize=(20,15))
            # plt.figure()
            for cat, hw in dict_cat_bbxhw.items():
                for bh,bw in hw:
                    plt.scatter(x=bw, y=bh,  marker='o', s=0.7*min(bw,bh))
            plt.legend(dict_cat_bbxhw.keys(),ncol=11)
            # plt.legend(dict_cat_bbxhw.keys(),ncol=11, bbox_to_anchor=(-0.03,-0.05), loc='upper left')
            # plt.legend(dict_cat_bbxhw.keys(),ncol=11, loc='lower right', labelspacing=0.2)
            plt.xlabel('边界框宽度', fontdict=fontdict)
            plt.ylabel('边界框高度', fontdict=fontdict)
            plt.title('类别边界框尺寸多样性', fontdict=title_fontdict)
            plt.savefig(os.path.join(source_dir, f'{sf}_cat_bbxhw.jpg'))
            plt.show()

            