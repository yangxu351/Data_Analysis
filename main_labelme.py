import argparse
from analysis_scripts.diversity.analysis_cls import ana_cls, stat_cls
from analysis_scripts.diversity.analysis_odt import ana_odt, stat_odt
from analysis_scripts.diversity.analysis_seg import ana_msk, stat_msk
from data_preprocess import coco2yolo
from data_preprocess import voc2yolo
from data_preprocess import coco2png
from data_preprocess import labelme2yolo
from analysis_scripts.correctness import cls_correct, odt_correct, seg_correct
from analysis_scripts.integrity import odt_integrity, seg_integrity
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
    parser.add_argument('--odt_lbl_format', type=str, default='labelme', help='[labelme, voc, yolo, coco]')
    parser.add_argument('--dst_split', type=str, default='test', help='[val, train, test]')
    parser.add_argument('--task', type=str, default='odt', help='[cls, odt, seg]')
    
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/Lunar', help='segmentation')
    # parser.add_argument('--seg_lbl_format', type=str, default='png', help='[coco, png]')
    # parser.add_argument('--dst_split', type=str, default='train', help='[val, train, test]')
    # parser.add_argument('--task', type=str, default='seg', help='[cls, odt, seg]')

    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/COCO', help='segmentation')
    # parser.add_argument('--src_split', type=str, default='val2017', help='[val2017, train2017]')
    # parser.add_argument('--dst_split', type=str, default='val', help='[val, train, test]')
    # parser.add_argument('--seg_lbl_format', type=str, default='coco', help='[coco, png]')
    # parser.add_argument('--task', type=str, default='seg', help='[cls, odt, seg]')
    
    parser.add_argument('--num_thres', type=int, default=3, help='类别样本数目阈值')
    parser.add_argument('--num_cats', type=int, default=80, help='类别样本总数')
    parser.add_argument('--wh_ratio_thres', type=int, default=20, help='w/h threshold')
    parser.add_argument('--img_thres', type=int, default=60, help='图像全黑像素的比例阈值')

    args = parser.parse_args()
    
    if args.task == 'cls':
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_classification.log'
        )
        cls_correct.outlier_unreadable_img_detection(args.base_dir, outlier_thresh=0.5, mapping_file=args.class_mapping_file, abnormal_condition=0)
        stat_cls(args.base_dir, category_imgnum_thresh=args.num_thres, mapping_file=args.class_mapping_file)
        ana_cls(args.base_dir)
    elif args.task == 'odt':
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_detection.log'
        )
        if args.odt_lbl_format == 'coco':
            coco2yolo.convert_coco_to_yolo(args.base_dir, src_split=args.src_split, split=args.dst_split)
            coco2yolo.move_imgs_to_specified_folder(args.base_dir, src_split=args.src_split, split=args.dst_split)
        elif args.odt_lbl_format == 'voc':
            voc2yolo.covert_voc_to_yolo(args.base_dir)
            voc2yolo.movefiles_by_setfile(args.base_dir, split=args.dst_split)
        elif args.odt_lbl_format == 'labelme':
            labelme2yolo.covert_labelme_to_yolo(args.base_dir)
            labelme2yolo.movefiles_by_setfile(args.base_dir, split=args.dst_split)
        odt_correct.outlier_unreadable_img_detection(args.base_dir, outlier_thresh=0.5, split=args.dst_split)
        odt_integrity.does_match_img_lbl(args.base_dir, split=args.dst_split)
        odt_correct.check_id(args.base_dir, num_cls=args.num_cats, split=args.dst_split)
        odt_correct.empty_check_lbl(args.base_dir, split=args.dst_split)
        odt_correct.check_coordinates(args.base_dir, wh_ratio_thres=args.wh_ratio_thres, split=args.dst_split)
        stat_odt(args.base_dir, category_imgnum_thresh=args.num_thres, split=args.dst_split)
        ana_odt(args.base_dir, split=args.dst_split)
    elif args.task == 'seg':
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_segmentation.log'
        )
        if args.seg_lbl_format == 'coco':
            coco2png.convert_coco_to_png(args.base_dir, src_split=args.src_split, split=args.dst_split)
            coco2png.move_imgs_to_specified_folder(args.base_dir, split=args.dst_split)
        seg_correct.outlier_unreadable_img_detection(args.base_dir, outlier_thresh=0.5, split=args.dst_split)
        seg_correct.empty_unreadable_check_msk_png(args.base_dir, split=args.dst_split)
        seg_integrity.does_match_img_lbl(args.base_dir, split=args.dst_split)
        seg_correct.check_msk_id(args.base_dir, num_cls=args.num_cats, split=args.dst_split)
        stat_msk(args.base_dir, category_imgnum_thresh=args.num_thres, split=args.dst_split)
        ana_msk(args.base_dir, split=args.dst_split)
    else:
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_inspection_others.log'
        )
        logging.warning('please input task!!!')
