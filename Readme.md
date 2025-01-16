# Data analysis


[Deep Learning Model Uncertainty in Medical Image Analysis](https://github.com/ddrevicky/deep-learning-uncertainty)

[龙石数据](https://www.longshidata.com/pages/quality.html)
## 要求
数据审查
1. 支持数据集准确性审查，提供异常数据检查、缺失值检查等功能;  :ok:
2. 支持数据集完整性审查，提供内容完整性检查、重复数据检查等功能; :ok:
3. 支持数据集量级、类别分布均衡性审查。 :ok:

### 准确性
1. 异常数据检查：
（1）图像内容全黑的像素个数不能超过图像总像素数量的50% :ok:
（2）标签类别id 必须小于 类别总数；:ok:
（3）目标检测标签（cx,cy,w,h）需要满足[0,1]之间  :ok:
（4）目标检测标签宽高比  :ok:
2. 图像可读性检查 :ok:
3. 标签缺失检查：标签内容是否为空 :ok:


### 完整性
1. 图像和标签是否一一匹配，图像、标签是否缺失 :ok: (目标检测VOC format，标签缺失在数据处理过程中)
2. 图像、标签是否有重复 :ok: (目标检测COCO format, VOC format，重复性检查在数据处理过程中)

### 测试用例——VOC （目标检测）
1. 坐标值为负       :ok:
2. 坐标值越界       :ok:
3. 边界框坐标重复    :ok:
4. 缺失边界框标注信息 :ok:
5. 宽高比异常       :ok:
6. 图像全黑像素超过50% :ok:
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
dataset
|__train
    |__category01
        |__2011_003876.png/*.jpg/*.bmp/*.tiff
        |__2211_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2211_003324.png/*.jpg/*.bmp/*.tiff
        |__2321_324089.png/*.jpg/*.bmp/*.tiff
    |__...
|__val
    |__category01
        |__2231_003876.png/*.jpg/*.bmp/*.tiff
        |__2232_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2234_003324.png/*.jpg/*.bmp/*.tiff
        |__2235_324089.png/*.jpg/*.bmp/*.tiff
    |__...
|__test
    |__category01
        |__3451_003876.png/*.jpg/*.bmp/*.tiff
        |__3461_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2543_003324.png/*.jpg/*.bmp/*.tiff
        |__2890_324089.png/*.jpg/*.bmp/*.tiff
    |__...
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
### YOLO 目录树
```
Dataset_YOLO
|__train
    |__images
        |__2011_003876.jpg/*.png/*.tiff/*.bmp
        |__....
    |__labels
        |__2011_003876.txt
        |__....
|__val
    |__images
        |__2231_003876.jpg/*.png/*.tiff/*.bmp
        |__....
    |__labels
        |__2231_003876.txt
        |__....
|__test
    |__images
        |__2561_003876.jpg/*.png/*.tiff/*.bmp
        |__....
    |__labels
        |__2561_003876.txt
        |__....
```
#### 其中，标注文件*.txt 满足归一化后YOLO标注格式:ID cw ch w h,以空格间隔,例如[ID center_x center_y width height] 即 [2 0.4046 0.8406 0.5031 0.2437]


### VOC 目录树
```
- Dataset_VOC
    - Annotations
        |__0d4c5e4f-105.xml
        |__0ddfc5aea-144.xml
        |__...
    - ImageSets
        |__Main
            |__train.txt
            |__test.txt
            |__val.txt
            |__trainval.txt
    - JPEGImages
        |__0d4c5e4f-105.jpg/*.png/*.tiff/*.bmp
        |__0ddfc5aea-144.jpg/*.png/*.tiff/*.bmp
        |__...
```

#### 其中，标注文件*.xml数据格式如下
```
<annotation>
    <folder>Dataset_VOC</folder>
    <filename>0d4c5e4f-105.jpg</filename>
    <source>
        <database>Unknown</database>
        <annotation>Unknown</annotation>
    </source>
    <size>
        <width>600</width>
        <height>400</height>
        <depth>3</depth>
    </size>
    <object>
        <name>cat</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>120</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>
```


### COCO 目录树
```
- Dataset_COCO
    - annotations
        - instances_train.json
        - instances_val.json
        - instances_test.json (可选)
    - images
        - train
            - 0d4c5e4f-fc3c-4d5a-906c-105.jpg
            - ...
        - val
            - 0ddfc5aea-fcdac-421-92dad-144.jpg
            - ...
        - test (可选)
            - 0hkkl5iop-fcdac-421-92fdl-453.jpg
            - ...
```

### labelme 数据格式
```
Dataset3
|__train
    |__images
        |__2011_003876.jpg/*.png/*.tiff/*.bmp
        |__....
    |__annotations
        |__2011_003876.json
        |__....
|__val
    |__images
        |__2231_003876.jpg/*.png/*.tiff/*.bmp
        |__....
    |__annotations
        |__2231_003876.json
        |__....
|__test
    |__images
        |__2561_003876.jpg/*.png/*.tiff/*.bmp
        |__....
    |__annotations
        |__2561_003876.json
        |__....
```

#### 其中，标注文件*.json数据格式如下
```
{
    "version": "4.5.6",
    "flags": {},
    "shapes": [
        {
            "label": "cat",
            "points": [
                [120.0, 100.0],
                [200.0, 100.0],
                [200.0, 200.0],
                [120.0, 200.0]
            ],
            "group_id": null,
            "shape_type": "rectangle",
            "flags": {}
        }
    ],
    "imagePath": "2231_003876.jpg",
    "imageData": null,
    "imageHeight": 400,
    "imageWidth": 600
}
```


### 目标检测数据说明  

目标检测的数据比分类复杂，一张图像中，需要标记出各个目标区域的位置和类别。

一般的目标区域位置用一个矩形框来表示，一般用以下3种方式表达：

|         表达方式    |                 说明               |
| :----------------: | :--------------------------------: |
|     x1,y1,x2,y2    | (x1,y1)为左上角坐标，(x2,y2)为右下角坐标  |  
|     x1,y1,w,h      | (x1,y1)为左上角坐标，w为目标区域宽度，h为目标区域高度  |
|     xc,yc,w,h    | (xc,yc)为目标区域中心坐标，w为目标区域宽度，h为目标区域高度  |  


 VOC采用的`[x1,y1,x2,y2]` 表示物体的bounding box

 COCO采用的`[x1,y1,w,h]` 表示物体的bounding box

 YOLO采用的`[xc,yc,w,h]`表示物体的bounding box


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
        |__2011_003876.jpg/*.png/*.tiff/*.bmp
        |__....
    |__masks
        |__2011_003876.jpg/*.png/*.tiff/*.bmp
        |__....
|__val
    |__images
        |__2121_003283.jpg/*.png/*.tiff/*.bmp
        |__....
    |__masks
        |__2121_003283.jpg/*.png/*.tiff/*.bmp
        |__....
|__test
    |__images
        |__2151_003345.jpg/*.png/*.tiff/*.bmp
        |__....
    |__masks
        |__2151_003345.jpg/*.png/*.tiff/*.bmp
        |__....    
```
#### 其中，masks文件夹下图像如果为灰度图像，像素值为该像素对应的类别索引
#### 如果masks文件夹下图像如果为RGB图像，像素值为该像素对应的颜色值(R,G,B),取值范围均在[0,255]
#### images文件夹下xyz.png 与masks文件夹下xyz.jpg 的名字要一致

