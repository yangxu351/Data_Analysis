# Data analysis


[Deep Learning Model Uncertainty in Medical Image Analysis](https://github.com/ddrevicky/deep-learning-uncertainty)

[龙石数据](https://www.longshidata.com/pages/quality.html)

## 图像分类（训练集、验证集、测试集）
### 类别多样性
1. 图像类别数量分布，即标签丰富性
2. 图像类别数量低于阈值的类别比率，即类别大小多样性
### 尺寸多样性
1. 图像尺寸数量分布

### 目录树
```
data
|__train
    |__c1
    |__c2
    |__c3
    ...
    |__cn
|__val
    |__c1
    |__c2
    |__c3
    ...
    |__cn
|__test
    |__c1
    |__c2
    |__c3
    ...
    |__cn
```
#### c* 表示类别文件夹，包含属于该类别的图像数据


## 目标检测 （训练集、验证集、测试集）
### 类别多样性
1. 实例类别数量分布，即标签丰富性
2. 实例类别所在的图像数量/数据集中图像总数，即相对标签丰度
3. 实例类别数量低于阈值的类别比率，即类别大小多样性
### 尺寸多样性
1. 实例尺寸数量分布
### 数据完整性
1. 图像中不包含目标实例的个数，即数据元素完整性

### 目录树
```
data
|__train
    |__images
        |__*.jpg/*.png/....
    |__labels
        |__*.txt
|__val
    |__images
        |__*.jpg/*.png/....
    |__labels
        |__*.txt
|__test
    |__images
        |__*.jpg/*.png/....
    |__labels
        |__*.txt
```
#### 其中*.txt 满足归一化后YOLO标注格式: ID cw ch w h
### VOC 数据格式
- `VOC2007/`
    - `Annotations/`
        - `0d4c5e4f-fc3c-4d5a-906c-105.xml`
        - `0ddfc5aea-fcdac-421-92dad-144.xml`
        - `...`
    - `ImageSets/`
        - `Main/`
            - `train.txt`
            - `test.txt`
            - `val.txt`
            - `trainval.txt`
    - `JPEGImages/`
        - `0d4c5e4f-fc3c-4d5a-906c-105.jpg`
        - `0ddfc5aea-fcdac-421-92dad-144.jpg`
        - `...`

### COCO 数据格式
- `coco/`
    - `annotations/`
        - `instances_train2017.json`
        - `instances_val2017.json`
    - `images/`
        - `train2017/`
            - `0d4c5e4f-fc3c-4d5a-906c-105.jpg`
            - `...`
        - `val2017`
            - `0ddfc5aea-fcdac-421-92dad-144.jpg`
            - `...`

## 语义分割数据目录树（训练集、验证集、测试集）
### 类别多样性
1. 类别数量分布，即标签丰富性
2. 类别所在的图像数量/数据集中图像总数，即相对标签丰度
3. 类别数量低于阈值的类别比率，即类别大小多样性
### 面积多样性
1. 实例类别像素面积分布

```
data
|__train
    |__images
        |__*.jpg/*.png/....
    |__masks
        |__*.jpg/*.png/....
|__val
    |__images
        |__*.jpg/*.png/....
    |__masks
        |__*.jpg/*.png/....
|__test
    |__images
        |__*.jpg/*.png/....
    |__masks
        |__*.jpg/*.png/....        
```
#### 其中，masks/XXX.jpg or XXX.png 为灰度图，像素值为该像素对应的类别索引