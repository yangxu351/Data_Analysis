import os
import sys 
from PIL import Image
import numpy as np
import json 
import matplotlib.pyplot as plt
from analysis_scripts.diversity.img_ana import caculate_brightness, calculate_image_contrast
import logging
logger = logging.getLogger(__name__)


plt.get_cmap('Set3')
title_fontdict = {'family' : 'Microsoft YaHei', 'size': 16}
fontdict = {"family": "SimHei", "size": 14}


def stat_cls(source_dir, dict_stat, category_imgnum_thresh=20, mapping_file=None):
    split_folders = ['train', 'val', 'test']
    if mapping_file is None:
        logging.error('class mapping file is needed!')
        sys.exit(1)
    dict_id_cat = json.load(open(mapping_file))
    classes = [x for x in dict_id_cat.values()]
    ids = [x for x in dict_id_cat.keys()]
    dict_cat_id = {classes[ix]:ids[ix] for ix in range(len(classes))}

    for sf in split_folders:
        if not os.path.isdir(os.path.join(source_dir, sf)):
            continue
        cat_folders = os.listdir(os.path.join(source_dir, sf))
        cat_folders.sort()
        
        brightness_avg_list = []
        contrast_list = []

        dict_cat_imgnum = {}
        dict_cat_imghw = {k:[] for k in dict_cat_id.keys()}
        for i, cf in enumerate(cat_folders):
            ori_imgs = os.listdir(os.path.join(source_dir, sf, cf))
            ori_imgs.sort()
            dict_cat_imgnum[cf] = len(ori_imgs)
            for name in ori_imgs:
                pil_img = Image.open(os.path.join(source_dir, sf, cf, name))
                try:
                    brightness = caculate_brightness(pil_img)
                except Exception as e:
                    logger.error(f"Unsupported image mode: {str(e)}", exc_info=True)
                brightness_avg_list.append(brightness)

                constrast = calculate_image_contrast(pil_img)
                contrast_list.append(constrast)

                img = np.array(pil_img)
                dict_cat_imghw[cf].append(img.shape[:2]) # h,w
        
        brightness_std = np.std(brightness_avg_list)
        contrast_std = np.std(contrast_list)

        dict_brightness = {'brightness_list': brightness_avg_list, 'brightness_std': brightness_std} 
        with open(os.path.join(source_dir, f'{sf}_allimg_volatility_brightness.json'), 'w') as f: # 亮度波动性
            json.dump(dict_brightness, f, indent=3)

        dict_contrast = {'constrast_list': contrast_list, 'contrast_std':contrast_std} 
        with open(os.path.join(source_dir, f'{sf}_allimg_volatility_contrast.json'), 'w') as f: # 对比度波动性
            json.dump(dict_contrast, f, indent=3)

        with open(os.path.join(source_dir, f'{sf}_cat_imgnum.json'), 'w') as f: #  类别样本数量分布
            json.dump(dict_cat_imgnum, f, ensure_ascii=False, indent=3)
        f.close()

        with open(os.path.join(source_dir, f'{sf}_cat_imghw.json'), 'w') as f: # 图像尺寸分布
            json.dump(dict_cat_imghw, f, ensure_ascii=False, indent=3)
        f.close()

        img_num = sum(dict_cat_imgnum.values())
        dict_stat['图像总数量'] = f'{img_num}'   # 图像总数量
        dabiao = 0
        for k, num in dict_cat_imgnum.items():
            dabiao += 1 if num > category_imgnum_thresh else 0
        dict_stat['类别样本数量达标率'] = f'{round(dabiao/len(dict_cat_imgnum.keys()),4)*100}%'    # 类别样本数量达标率

        dict_rel_lbl_diver = {k:1.*v/img_num for k,v in dict_cat_imgnum.items()}
        with open(os.path.join(source_dir, f'{sf}_relative_cat_diversity.json'), 'w') as f: # 相对类别多样性
            json.dump(dict_rel_lbl_diver, f, ensure_ascii=False, indent=3)
        f.close() 

        dict_lbl_size_diver = {k: 1 if v<category_imgnum_thresh else 0 for k,v in dict_cat_imgnum.items()}
        lbl_size_diver = 1.*sum(dict_lbl_size_diver.values())/len(dict_lbl_size_diver.keys())
        with open(os.path.join(source_dir, f'{sf}_cat_num_diversity.txt'), 'w') as f: # 类别大小多样性
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

    
    
