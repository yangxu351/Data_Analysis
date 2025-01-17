import os
import sys 
from PIL import Image
import numpy as np
import json 
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


plt.get_cmap('Set3')
title_fontdict = {'family' : 'Microsoft YaHei', 'size': 16}
fontdict = {"family": "SimHei", "size": 14}


def caculate_brightness(image):
    if image.mode == 'RGB':  # 彩色图像
        # 将图像转换为 numpy 数组并计算每个通道的平均亮度
        np_image = np.array(image)
        r = np_image[:, :, 0]
        g = np_image[:, :, 1]
        b = np_image[:, :, 2]
        avg_brightness = (np.mean(r) + np.mean(g) + np.mean(b)) / 3
    elif image.mode == 'L':  # 灰度图像
        np_image = np.array(image)
        avg_brightness = np.mean(np_image)
    else:
        logger.error("Unsupported image mode!")
        sys.exit(1)
    return avg_brightness


def calculate_image_color_stats(image):
    image = image.convert('RGB')
    np_image = np.array(image)
    mean_rgb = np.mean(np_image, axis=(0, 1))  # 计算整个图像的 RGB 通道均值
    std_rgb = np.std(np_image, axis=(0, 1))    # 计算整个图像的 RGB 通道标准差
    return mean_rgb, std_rgb


def calculate_overall_color_volatility(image_paths):
    all_mean_rgb = []
    all_std_rgb = []
    for path in image_paths:
        mean_rgb, std_rgb = calculate_image_color_stats(path)
        all_mean_rgb.append(mean_rgb)
        all_std_rgb.append(std_rgb)
    all_mean_rgb = np.array(all_mean_rgb)
    all_std_rgb = np.array(all_std_rgb)
    overall_mean_rgb = np.mean(all_mean_rgb, axis=0)  # 计算所有图像的 RGB 通道均值的均值
    overall_std_rgb = np.mean(all_std_rgb, axis=0)  # 计算所有图像的 RGB 通道标准差的均值
    return overall_mean_rgb, overall_std_rgb


def stat_cls(source_dir, label_imgnum_thresh=20, mapping_file=None):
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
        rgb_avg_list = []
        rgb_std_list = []

        dict_cat_imgnum = {}
        dict_cat_imghw = {k:[] for k in dict_cat_id.keys()}
        for i, cf in enumerate(cat_folders):
            ori_imgs = os.listdir(os.path.join(source_dir, sf, cf))
            ori_imgs.sort()
            dict_cat_imgnum[cf] = len(ori_imgs)
            for name in ori_imgs:
                pil_img = Image.open(os.path.join(source_dir, sf, cf, name))
                brightness = caculate_brightness(pil_img)
                brightness_avg_list.append(brightness)

                mean_rgb, std_rgb = calculate_image_color_stats(pil_img)
                rgb_avg_list.append(mean_rgb)
                rgb_std_list.append(std_rgb)

                img = np.array(pil_img)
                dict_cat_imghw[cf].append(img.shape[:2]) # h,w
        
        brightness_std = np.std(brightness_avg_list)

        # overall_mean_rgb = np.mean(np.array(all_mean_rgb), axis=0)  # 计算所有图像的 RGB 通道均值的均值
        # overall_std_rgb = np.mean(np.array(all_std_rgb), axis=0)  # 计算所有图像的 RGB 通道标准差的均值

        dict_brightness = {'brightness_list': brightness_avg_list, 'brightness_std': brightness_std} 
        with open(os.path.join(source_dir, f'{sf}_allimg_volatility_brightness.json'), 'w') as f: # 亮度波动性
            json.dump(dict_brightness, f, indent=4)

        dict_color = {'color_volatility_list': rgb_avg_list, 'color_volatility_std': rgb_std_list} 
        with open(os.path.join(source_dir, f'{sf}_allimg_volatility_color.json'), 'w') as f: # 颜色波动性
            json.dump(dict_color, f, indent=4)

        with open(os.path.join(source_dir, f'{sf}_cat_imgnum.json'), 'w') as f: #  类别样本数量分布
            json.dump(dict_cat_imgnum, f, ensure_ascii=False, indent=3)
        f.close()

        with open(os.path.join(source_dir, f'{sf}_cat_imghw.json'), 'w') as f: # 图像尺寸分布
            json.dump(dict_cat_imghw, f, ensure_ascii=False, indent=3)
        f.close()

        img_num = sum(dict_cat_imgnum.values()) # 图像数量级
        with open(os.path.join(source_dir, f'{sf}_all_img_num.txt'), 'w') as f: 
            f.write(f'{img_num}%\n')
        f.close() 

        dict_rel_lbl_diver = {k:1.*v/img_num for k,v in dict_cat_imgnum.items()}
        with open(os.path.join(source_dir, f'{sf}_relative_cat_diversity.json'), 'w') as f: # 相对类别多样性
            json.dump(dict_rel_lbl_diver, f, ensure_ascii=False, indent=3)
        f.close() 

        dict_lbl_size_diver = {k: 1 if v<label_imgnum_thresh else 0 for k,v in dict_cat_imgnum.items()}
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

    
    
