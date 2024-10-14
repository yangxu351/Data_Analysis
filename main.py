import argparse
from analysis_scripts.analysis_cls import ana_cls, stat_cls
from analysis_scripts.analysis_odt import ana_odt, stat_odt
from analysis_scripts.analysis_seg import ana_msk, stat_msk



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/MSTAR', help='classification')
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/COCO', help='object detection')
    # parser.add_argument('--lbl_format', type=str, default='coco', help='[coco, voc, yolo]')
    parser.add_argument('--src_split', type=str, default='val2017', help='[val2017, train2017]')
    parser.add_argument('--dst_split', type=str, default='val', help='[val, train, test]')
    parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/Lunar', help='segmentation')
    parser.add_argument('--num_thres', type=int, default=20, help='类别样本数目阈值')
    parser.add_argument('--task', type=str, default='seg', help='[cls, odt, seg]')
    parser.add_argument('--img_suffix', type=str, default='.jpg', help='.png .jpg')
    parser.add_argument('--lbl_suffix', type=str, default='.txt', help='.txt')
    args = parser.parse_args()
    
    if args.task == 'cls':
        stat_cls(args.base_dir,label_imgnum_thresh=args.num_thres)
        ana_cls(args.base_dir)
    elif args.task == 'odt':
        if args.lbl_format == 'coco':
            from data_preprocess import coco2yolo
            coco2yolo.convert_coco_to_yolo(args.base_dir, src_split=args.src_split, split=args.dst_split)
            coco2yolo.move_imgs_to_specified_folder(args.base_dir, src_split=args.src_split, split=args.dst_split)
        elif args.lbl_format == 'voc':
            from data_preprocess import voc2yolo
            voc2yolo.covert_voc_to_yolo(args.base_dir)
            voc2yolo.movefiles_by_setfile(args.base_dir, split=args.dst_split)

        # else: # yolo

        stat_odt(args.base_dir,label_imgnum_thresh=args.num_thres)
        ana_odt(args.base_dir)
    elif args.task == 'seg':
        stat_msk(args.base_dir, label_imgnum_thresh=args.num_thres)
        ana_msk(args.base_dir)
    else:
        print('please input task!!!')
