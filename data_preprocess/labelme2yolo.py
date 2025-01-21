import os
import shutil 
import pandas as pd
import numpy as np
import json 
import logging
logger = logging.getLogger(__name__)


def make_folder_if_not(dst_dir, rm_exist=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        if rm_exist:
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)


def convert_labelme_to_yolo(labelme_json_path, output_dir):
    # 类别名称到类别 ID 的映射，需要根据实际情况修改
    class_mapping = {"person": 0, "car": 1, "dog": 2}  # 示例类别映射，可按需添加更多类别

    with open(labelme_json_path, 'r') as f:
        labelme_data = json.load(f)

    image_width = labelme_data["imageWidth"]
    image_height = labelme_data["imageHeight"]
    shapes = labelme_data["shapes"]

    yolo_annotations = []
    for shape in shapes:
        label = shape["label"]
        class_id = class_mapping.get(label, -1)  # 获取类别 ID，如果类别不在映射中则为 -1
        if class_id == -1:
            continue  # 跳过未在映射中的类别

        points = shape["points"]
        x_min = min([point[0] for point in points])
        y_min = min([point[1] for point in points])
        x_max = max([point[0] for point in points])
        y_max = max([point[1] for point in points])

        # 坐标转换为 YOLO 格式
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    image_filename = os.path.basename(labelme_data["imagePath"]).split('.')[0]
    yolo_txt_path = os.path.join(output_dir, image_filename + ".txt")
    with open(yolo_txt_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(annotation + '\n')



def covert_labelme_to_yolo(base_dir, dict_stat, wh_thres=50, mapping_file='', extention='.json'):
    ''' 
        convert labelme to YOLO 
        empty labels
        duplicate annotations
    '''
    json_dir = os.path.join(base_dir, 'annotations')
    txt_dir = os.path.join(base_dir, 'labels') 
    make_folder_if_not(txt_dir)

    json_files = os.listdir(json_dir)
    dict_id_cat = json.load(open(mapping_file))
    classes = [x for x in dict_id_cat.values()]
    ids = [x for x in dict_id_cat.keys()]
    dict_cat_id = {classes[ix]:ids[ix] for ix in range(len(classes))}

    index = 0
    dict_duplicate_file_box = {}
    empty_lbls = []
    dict_invalid_bbx_jsons = {}
    suffix_error_files = []
    attribute_lost_jsons = []
    invalid_cat_files = []
    
    for file_name in json_files:
        if not file_name.endswith(extention):
            suffix_error_files.append(file_name)
            continue
        xf = os.path.join(json_dir, file_name)
        with open(xf, 'r') as f:
            labelme_data = json.load(f)
        try:
            image_width = labelme_data["imageWidth"]
            image_height = labelme_data["imageHeight"]
            shapes = labelme_data["shapes"]
        except:
            logger.warning('key attribute lost!')
            attribute_lost_jsons.append(file_name)
            continue 

        yolo_annotations = []
        for shape in shapes:
            try:
                label = shape["label"]
            except:
                logger.warning('json without attribute-->label ')
                empty_lbls.append(file_name)
                continue
            class_id = dict_cat_id.get(label, -1)  # 获取类别 ID，如果类别不在映射中则为 -1
            if class_id == -1:
                invalid_cat_files.append(file_name)
                continue  # 跳过未在映射中的类别

            try:
                points = shape["points"]
            except:
                logger.warning('json without attribute-->points ') 
                if file_name not in dict_invalid_bbx_jsons.keys():
                    dict_invalid_bbx_jsons[file_name]= [[class_id, None, None, None, None]]
                else:
                    dict_invalid_bbx_jsons[file_name].append([class_id, None, None, None, None])
                continue
                
            x_min = min([point[0] for point in points])
            y_min = min([point[1] for point in points])
            x_max = max([point[0] for point in points])
            y_max = max([point[1] for point in points])

            width = x_max - x_min
            height = y_max - y_min
            # check invlid coordinates
            if x_min < 0 or y_min < 0 or x_max >image_width or y_max >image_height or width<=0 or height<=0 or (max(width/height, height/width) > wh_thres):
                if file_name not in dict_invalid_bbx_jsons.keys():
                    dict_invalid_bbx_jsons[file_name]= [[class_id, x_min, y_min, x_max, y_max]]
                else:
                    dict_invalid_bbx_jsons[file_name].append([class_id, x_min, y_min, x_max, y_max])
                continue

            # 坐标转换为 YOLO 格式
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            box = [class_id, x_center, y_center, width, height]
            # check duplicate
            if box not in yolo_annotations:
                yolo_annotations.append(box)
            else:
                if file_name not in dict_duplicate_file_box.keys():
                    dict_duplicate_file_box[file_name] = np.array(box, dtype=np.int32).tolist()
                else:
                    dict_duplicate_file_box[file_name].append(np.array(box, dtype=np.int32).tolist())
            
        lbl_file = os.path.join(txt_dir, os.path.basename(xf).replace(extention, '.txt'))
        with open(lbl_file, 'w') as f:
            for cat_id, cx,cy,w,h in yolo_annotations:
                f.write("%d %.4f %.4f %.4f %.4f\n" % (cat_id, cx, cy, w, h))
        f.close() 

    error_ext_file = os.path.join(base_dir, 'all_jsons_error_extention.txt')
    with open(error_ext_file, 'w') as f:
        for name in suffix_error_files:
            f.write("%s\n" % name)
    f.close()
    dict_stat['标注文件格式不一致比例'] = f'{round(len(suffix_error_files)/len(json_files),4)*100}%'   # 标注文件格式不一致比例

    attribute_lost_file = os.path.join(base_dir, 'all_jsons_lost_key_attributes.txt')
    with open(attribute_lost_file, 'w') as f:
        for name in attribute_lost_jsons:
            f.write("%s\n" % name)
    f.close()  
    dict_stat['标注关键字段缺失率'] = f'{round(len(attribute_lost_jsons)/len(json_files),4)*100}%'   # 标注关键字段缺失率

    file_invalid_xml = os.path.join(base_dir, 'all_jsons_invalid_coords.json')
    with open(file_invalid_xml, 'w') as f:
        json.dump(dict_invalid_bbx_jsons, f, ensure_ascii=False, indent=3)
    f.close()   
    dict_stat['标注不合理比例'] = f'{round(len(dict_invalid_bbx_jsons.keys())/len(json_files),4)*100}%' # 标注不合理比例

    file_empty = os.path.join(base_dir, 'all_jsons_without_lbls.txt')
    with open(file_empty, 'w') as f:
        for name in empty_lbls:
            f.write("%s\n" % name)
    f.close()   
    dict_stat['空标注比例'] = f'{round(len(empty_lbls)/len(json_files),4)*100}%'  # 空标注比例

    with open(os.path.join(base_dir, 'all_jsons_duplicate_bbox.json'), 'w') as f: # dict of file_name:[bbox]
        json.dump(dict_duplicate_file_box, f, ensure_ascii=False, indent=3)
    f.close() 
    dict_stat['标注重复比例'] = f'{round(len(dict_duplicate_file_box.keys())/len(json_files),4)*100}%'  # 空标注比例

def movefiles_by_setfile(base_dir, split='train'):
    '''
    create new folder and move files to specified folder according to the set file
    '''
    set_dir = os.path.join(base_dir, 'ImageSets/Main')
    img_dir = os.path.join(base_dir, 'JPEGImages')
    txt_dir = os.path.join(base_dir, 'labelsAll') 

    set_file = os.path.join(set_dir, f'{split}.txt')
    df_set = pd.read_csv(set_file, header=None)
    img_foler = os.path.join(base_dir, split, 'images')
    make_folder_if_not(img_foler)
    lbl_foler = os.path.join(base_dir, split, 'labels')
    make_folder_if_not(lbl_foler)
    for name in df_set.iloc[:,0]:
        img_name = f'{name}.jpg'
        lbl_name = f'{name}.txt'
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(img_foler, img_name))
        shutil.copy(os.path.join(txt_dir, lbl_name), os.path.join(lbl_foler, lbl_name))

if __name__ == "__main__":
    base_dir = 'F:/Public_Dataset/BCCD_VOC'

    
    covert_labelme_to_yolo(base_dir)

    
    movefiles_by_setfile(base_dir, split='train')
    movefiles_by_setfile(base_dir, split='val')
    movefiles_by_setfile(base_dir, split='test')
    
    
        
