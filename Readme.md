# Data analysis


[Deep Learning Model Uncertainty in Medical Image Analysis](https://github.com/ddrevicky/deep-learning-uncertainty)

[龙石数据](https://www.longshidata.com/pages/quality.html)
## 要求
数据审查
1. 支持数据集准确性审查，提供异常数据检查、缺失值检查等功能;  :ok:
2. 支持数据集完整性审查，提供内容完整性检查、重复数据检查等功能; :ok:
3. 支持数据集量级、类别分布均衡性审查。:ok:

### 准确性
1. 异常数据检查：
（1）图像内容全黑的像素个数不能超过图像总像素数量的50% :ok:
（2）标签类别id 必须小于 类别总数；:ok:
（3）目标检测标签（cx,cy,w,h）需要满足[0,1]之间  :ok:
（4）目标检测标签宽高比  :ok:
2. 图像可读性检查 :ok:
3. 标签缺失检查：标签内容是否为空 :ok:


### 完整性
1. 图像和标签是否一一匹配，图像、标签是否缺失:ok: (目标检测VOC format，标签缺失在数据处理过程中)
2. 图像、标签是否有重复 :ok: (目标检测COCO format, VOC format，重复性检查在数据处理过程中)

### 测试用例——VOC （目标检测）
1. 坐标值为负       :ok:
2. 坐标值越界       :ok:
3. 边界框坐标重复    :ok:
4. 缺失边界框标注信息:ok:
5. 宽高比异常       :ok:
6. 图像全黑像素超过50%:ok:
7. 图像与标签不匹配 :ok:

### 测试用例--COCO （目标检测）
1. 坐标值为负           :ok:
2. 坐标值越界           :ok:
3. 边界框类别id越界     :ok:
4. 类别信息重复         :ok:
5. 边界框坐标重复       :ok:
6. 缺失边界框标注信息   :ok:
7. 宽高比异常           :ok:
8. 图像全黑像素超过60%  :ok:
9. 图像与标签不匹配     :ok:

### 测试用例--COCO （语义分割）
1. 分割类别id越界   :ok:  
2. 分割标注信息重复 :ok:
3. 分割标注信息缺失 :ok:
4. 分割标注信息越界 :ok:
5. 图像全黑像素超过60%  :ok:
6. 图像与标签不匹配     :ok:

### 测试用例--Lunar PNG （语义分割）
1. 分割类别id越界   :ok:  
2. 图像全黑像素超过60%  :ok:
3. 图像与标签不匹配     :ok:

## 图像分类（训练集、验证集、测试集）
### 类别多样性
1. 图像类别数量分布，即标签丰富性  :ok:
2. 图像类别数量低于阈值的类别比率，即类别大小多样性  :ok:
### 尺寸多样性
1. 图像尺寸数量分布  :ok:

### 目录树
```
data
|__train
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
|__val
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
|__test
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
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
        |__*.jpg/*.png/*.tiff/*.bmp/....
    |__labels
        |__*.txt
|__val
    |__images
        |__*.jpg/*.png/*.tiff/*.bmp/....
    |__labels
        |__*.txt
|__test
    |__images
        |__*.jpg/*.png/*.tiff/*.bmp/....
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
        |__*.jpg/*.png/*.tiff/*.bmp/....
    |__masks
        |__*.jpg/*.png/*.tiff/*.bmp/....
|__val
    |__images
        |__*.jpg/*.png/*.tiff/*.bmp/....
    |__masks
        |__*.jpg/*.png/*.tiff/*.bmp/....
|__test
    |__images
        |__*.jpg/*.png/*.tiff/*.bmp/....
    |__masks
        |__*.jpg/*.png/*.tiff/*.bmp/....        
```
#### 其中，masks/XXX.jpg or XXX.png 为灰度图，像素值为该像素对应的类别索引