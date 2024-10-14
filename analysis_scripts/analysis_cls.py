import os
from PIL import Image
import numpy as np
import json 
import matplotlib.pyplot as plt
plt.get_cmap('Set3')
title_fontdict = {'family' : 'Microsoft YaHei', 'size': 16}
fontdict = {"family": "SimHei", "size": 14}

def stat_cls(source_dir, label_imgnum_thresh=20):
    split_folders = ['train', 'val', 'test']
    for sf in split_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        cat_folders = os.listdir(os.path.join(source_dir, sf))
        cat_folders.sort()
        cat_dict = {}
        for i, c in enumerate(cat_folders):
            cat_dict[c] = i
        with open(os.path.join(source_dir, f'{sf}_cat_id_dict.json'), 'w') as f:
            json.dump(cat_dict, f, ensure_ascii=False, indent=3)
        f.close()

        dict_cat_imgnum = {}
        dict_cat_imghw = {k:[] for k in cat_dict.keys()}
        for i, cf in enumerate(cat_folders):
            ori_imgs = os.listdir(os.path.join(source_dir, sf, cf))
            ori_imgs.sort()
            dict_cat_imgnum[cf] = len(ori_imgs)
            for name in ori_imgs:
                img = np.array(Image.open(os.path.join(source_dir, sf, cf, name)))
                dict_cat_imghw[cf].append(img.shape[:2]) # h,w
        
        with open(os.path.join(source_dir, f'{sf}_cat_imgnum.json'), 'w') as f: #  类别多样性
            json.dump(dict_cat_imgnum, f, ensure_ascii=False, indent=3)
        f.close()

        with open(os.path.join(source_dir, f'{sf}_cat_imghw.json'), 'w') as f: # 图像尺寸多样性
            json.dump(dict_cat_imghw, f, ensure_ascii=False, indent=3)
        f.close()

        img_num = sum(dict_cat_imgnum.values())
        dict_rel_lbl_diver = {k:1.*v/img_num for k,v in dict_cat_imgnum.items()}
        with open(os.path.join(source_dir, f'{sf}_relative_cat_diversity.json'), 'w') as f: # 相对类别多样性
            json.dump(dict_rel_lbl_diver, f, ensure_ascii=False, indent=3)
        f.close() 

        dict_lbl_size_diver = {k: 1 if v<label_imgnum_thresh else 0 for k,v in dict_cat_imgnum.items()}
        lbl_size_diver = 1.*sum(dict_lbl_size_diver.values())/len(dict_lbl_size_diver.keys())
        with open(os.path.join(source_dir, f'{sf}_cat_size_diversity.txt'), 'w') as f: # 类别大小多样性
            f.write(f'{lbl_size_diver*100}%\n')
        f.close() 

def ana_cls(source_dir):
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

            dict_imghw = json.load(open(os.path.join(source_dir, f'{sf}_cat_imghw.json')))
            plt.figure()
            for k,hw_list in dict_imghw.items():
                for hw in hw_list:
                    plt.scatter(x=hw[1], y=hw[0],  marker='o')
            plt.legend(dict_imghw.keys())
            plt.xlabel('图片宽度', fontdict=fontdict)
            plt.ylabel('图片高度', fontdict=fontdict)
            plt.title('图片尺寸多样性', fontdict=title_fontdict)
            plt.savefig(os.path.join(source_dir, f'{sf}_cat_imghw.jpg'))
            plt.show()

    
    
