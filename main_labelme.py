import argparse
import os 
import json 
from analysis_scripts.diversity.analysis_cls import  stat_cls
from analysis_scripts.diversity.analysis_odt import  stat_odt
from analysis_scripts.diversity.analysis_seg import  stat_msk
from data_preprocess import coco2yolo
from data_preprocess import voc2yolo
from data_preprocess import coco2png
from data_preprocess import labelme2yolo
from data_preprocess import yolo
from analysis_scripts.correctness import cls_correct, odt_correct, seg_correct
from analysis_scripts.integrity import seg_integrity
import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/MSTAR', help='classification')
    # parser.add_argument('--task', type=str, default='cls', help='[cls, odt, seg]')
    # parser.add_argument('--class_mapping_file', type=str, default='F:/Public_Dataset/MSTAR/class.json', help='id class mapping file')


    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/COCO', help='object detection')
    # parser.add_argument('--src_split', type=str, default='val2017', help='[val2017, train2017]')
    # parser.add_argument('--dst_split', type=str, default='val', help='[val, train, test]')
    # parser.add_argument('--odt_lbl_format', type=str, default='coco', help='[labelme, coco, voc, yolo]')
    # parser.add_argument('--task', type=str, default='odt', help='[cls, odt, seg]')
    
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/BCCD_VOC', help='object detection')
    # parser.add_argument('--class_mapping_file', type=str, default='F:/Public_Dataset/BCCD_VOC/class.json', help='id class mapping file')
    # parser.add_argument('--odt_lbl_format', type=str, default='voc', help='[labelme, voc, yolo, coco]')
    # parser.add_argument('--dst_split', type=str, default='test', help='[val, train, test]')
    # parser.add_argument('--task', type=str, default='odt', help='[cls, odt, seg]')

    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/BCCD_VOC', help='object detection')
    # parser.add_argument('--class_mapping_file', type=str, default='F:/Public_Dataset/BCCD_VOC/class.json', help='id class mapping file')
    parser.add_argument('--odt_lbl_format', type=str, default='yolo', help='[labelme, voc, yolo, coco]')
    parser.add_argument('--dst_split', type=str, default='test', help='[val, train, test]')
    parser.add_argument('--task', type=str, default='odt', help='[cls, odt, seg]')
    
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/Lunar', help='segmentation')
    # parser.add_argument('--seg_lbl_format', type=str, default='png', help='[coco, png]')
    # parser.add_argument('--dst_split', type=str, default='train', help='[val, train, test]')
    # parser.add_argument('--task', type=str, default='seg', help='[cls, odt, seg]')
    # parser.add_argument('--class_mapping_file', type=str, default='F:/Public_Dataset/Lunar/class.json', help='id class mapping file')

    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/COCO', help='segmentation')
    # parser.add_argument('--src_split', type=str, default='val2017', help='[val2017, train2017]')
    # parser.add_argument('--dst_split', type=str, default='val', help='[val, train, test]')
    # parser.add_argument('--seg_lbl_format', type=str, default='coco', help='[coco, png]')
    # parser.add_argument('--task', type=str, default='seg', help='[cls, odt, seg]')
    
    
    parser.add_argument('--num_per_cat_thres', type=int, default=100, help='类别样本数目阈值')
    parser.add_argument('--num_cats', type=int, default=80, help='数据集类别总数’')
    parser.add_argument('--wh_ratio_thres', type=int, default=20, help='w/h threshold')
    parser.add_argument('--img_outlier_thres', type=float, default=0.5, help='图像全黑像素的比例阈值')

    args = parser.parse_args()
    
    if args.task == 'cls':
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_classification.log'
        )
        dict_stat = {}
        cls_correct.outlier_unreadable_img_detection(args.base_dir, dict_stat, outlier_thresh=args.img_outlier_thres, mapping_file=args.class_mapping_file, abnormal_condition=0)
        stat_cls(args.base_dir, dict_stat, category_imgnum_thresh=args.num_per_cat_thres, mapping_file=args.class_mapping_file)
        # ana_cls(args.base_dir)
        with open(os.path.join(args.base_dir, 'cls_static_values.json'), 'w') as f: 
            json.dump(dict_stat, f, ensure_ascii=False, indent=4)
        f.close() 
    elif args.task == 'odt':
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_detection.log'
        )
        dict_stat = {}
        if args.odt_lbl_format == 'coco':
            coco2yolo.convert_coco_to_yolo(args.base_dir, dict_stat, src_split=args.src_split, split=args.dst_split, wh_thres=args.wh_ratio_thres)
            coco2yolo.move_imgs_to_specified_folder(args.base_dir, src_split=args.src_split, split=args.dst_split)
        elif args.odt_lbl_format == 'voc':
            voc2yolo.covert_voc_to_yolo(args.base_dir, dict_stat, wh_thres=args.wh_ratio_thres, mapping_file=args.class_mapping_file)
            voc2yolo.dose_match_img_lbl(args.base_dir, dict_stat)
            voc2yolo.movefiles_by_setfile(args.base_dir, split=args.dst_split)
        elif args.odt_lbl_format == 'labelme':
            labelme2yolo.covert_labelme_to_yolo(args.base_dir, dict_stat, wh_thres=args.wh_ratio_thres, mapping_file=args.class_mapping_file)
            labelme2yolo.dose_match_img_lbl(args.base_dir, dict_stat)
        else: # 'yolo'
            yolo.check_txt(args.base_dir, dict_stat, mapping_file=args.class_mapping_file, split=args.dst_split)
            yolo.dose_match_img_lbl(args.base_dir, dict_stat)
        odt_correct.outlier_unreadable_img_detection(args.base_dir, dict_stat, outlier_thresh=args.img_outlier_thres, split=args.dst_split)
        stat_odt(args.base_dir, category_imgnum_thresh=args.num_per_cat_thres, split=args.dst_split)
        # ana_odt(args.base_dir, split=args.dst_split)
        with open(os.path.join(args.base_dir, 'odt_static_values.json'), 'w') as f: # dict of file_name:[bbox]
            json.dump(dict_stat, f, ensure_ascii=False, indent=4)
        f.close() 
    elif args.task == 'seg':
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_segmentation.log'
        )
        dict_stat = {}
        if args.seg_lbl_format == 'coco':
            coco2png.convert_coco_to_png(args.base_dir, dict_stat, src_split=args.src_split, split=args.dst_split)
            coco2png.move_imgs_to_specified_folder(args.base_dir, split=args.dst_split)
        seg_correct.outlier_unreadable_img_detection(args.base_dir, dict_stat, outlier_thresh=args.img_outlier_thres, split=args.dst_split)
        seg_correct.empty_unreadable_check_msk_png(args.base_dir, dict_stat, split=args.dst_split)
        seg_integrity.does_match_img_lbl(args.base_dir, dict_stat, split=args.dst_split)
        seg_correct.check_msk_id(args.base_dir, dict_stat, num_cls=args.num_cats, split=args.dst_split)
        stat_msk(args.base_dir, dict_stat, category_imgnum_thresh=args.num_per_cat_thres, split=args.dst_split)
        # ana_msk(args.base_dir, split=args.dst_split)
        with open(os.path.join(args.base_dir, 'seg_static_values.json'), 'w') as f: 
            json.dump(dict_stat, f, ensure_ascii=False, indent=4)
        f.close() 
    else:
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_others.log'
        )
        logging.warning('please input task!!!')
