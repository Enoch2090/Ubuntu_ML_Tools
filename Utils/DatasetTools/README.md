# DatasetTools工具使用说明

[TOC]

代码版本：v2.0.0

说明版本：v1.0.0

DatasetTools.ipynb中包含了`annotation`类的代码以及所有可用操作的示例代码。在需要对数据集的标注文件作出操作时，可以拷贝一份`DatasetTools.ipynb`文件到工作目录并直接在Jupyter Notebook里操作，也可以编写脚本，调用这个类。

## 准备工作

如在实验室电脑上工作，无需任何额外安装。如在其他设备上进行操作，需要安装对应的Python库。安装方法：

```shell
pip3 install -r requirements.txt
```

建议直接拷贝一份`DatasetTools.ipynb`文件到工作目录，在这个笔记本里操作，这样无需`import`引入。如使用`.py`脚本则需要拷贝``DatasetTools.py`到同级目录用于`import`，但是复用性更好。

## 使用方法

本节给出几个常用操作的示例，对于各方法的详细说明见第三节。

1. 仅合并几个已有的标注：

   ```python
   import DatasetTools as dt # 如在DatasetTools.ipynb里操作无需import，
   													# 直接 newSet = annotation()即可。
   newSet = dt.annotation()  # 完全使用默认配置新建一个空对象
   newSet.addSubset(annotPath="knife_1_task1/instances_default.json")
   newSet.addSubset(annotPath="knife_1_task2/instances_default.json")
   newSet.addSubset(annotPath="knife_1_task3/instances_default.json")
   newSet.save(fileName="knife_1_all.json", replace=True)
   ```

2. 合并标注+文件

   ```shell
   Suspicious_bottle/instances_all.json, Suspicious_bottle/all_Suspicious_bottles
   knife/instances_all.json, knife/all_knives
   firearms/instances_all.json, firearms/all_firearms
   ```

   保存为`list.txt`。

   ```python
   import DatasetTools as dt
   newSet = dt.annotation()
   newSet.addBatch(fileName="list.txt", inplace=True)
   ```

3. 对已有数据集执行分割。

   ```python
   import DatasetTools as dt
   newSet = dt.annotation()
   newSet.addSubset(annotPath="knife_1_all.json")
   newSet.split(train=9, test=1, val=1) # 按9:1:1进行分割
   ```

## 接口说明

本节给出DatasetTools的接口说明，方便自行编程使用。

DatasetTools的代码主要由`annotation`类及其对应的方法构成。

### annotation

```python
class DatasetTools.annotation(name="SentryDataSet", version="2.0", url="", annotPath="", imgPath="")
```

`annotation`类是对标注文件内容的抽象，包含了标注的类别、图片、具体标注等信息。

**参数：**

- **name:** string
  数据集的名字，默认为`SentryDataSet`。决定了默认输出的文件夹名称，以及显示在.json标注文件信息中的名称。

- **version:** string
  数据集的版本，默认为`2.0`。决定了默认输出的文件夹名称，以及显示在.json标注文件信息中的版本号。

- **url:** string
  数据集的链接，默认留空。决定显示在.json标注文件信息中的数据集链接，建议在Notion上新建页面备注对应数据集的详细信息，然后填写在此。

- **annotPath:** string
  初始化所用的标注文件的路径，默认留空。若留空则初始化一个空对象。如果目标是合并数个数据集，建议留空；如果目标是读取一个已有数据集并进行操作，建议填该数据集的标注文件。

- **imgPath:** string

  初始化所用的数据集图片的路径，默认留空。若留空则初始化一个空对象。如果目标是合并数个数据集，建议留空；如果目标是读取一个已有数据集并进行操作，建议填该数据集的图片路径。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()  // 完全使用默认配置新建一个空对象
```

### addSubset 

```python
def addSubset(annotPath = "instances_all.json", imagePath = "") 
```

向已有的`annotation`对象添加一个数据集子集。

**参数：**

- **annotPath:** string
  添加的数据集标注文件的路径。默认为`"instances_all.json"`，这个参数必须给出。
- **imagePath:** string
  添加的数据集图片的路径。默认留空，这个参数可以不填。**在合并的时候，必须保证**：
  - **每次调用`addSubset`时都给出**添加的数据集图片路径。这样在保存时，`save`方法会自动从各个路径合并图片。或者：
  - **每次调用`addSubset`时都不给出**添加的数据集图片路径。这样在保存时，`save`方法会跳过合并图片的操作。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()  # 完全使用默认配置新建一个空对象
newSet.addSubset(annotPath="knife_1.json", imagePath="knife/all_knives")
```

### addBatch

```python
def addBatch(fileName, inplace=False)
```

从`fileName`指向的文件读取一个列表。通过调用`addSubset`，向已有的`annotation`对象批量添加列表给出的数据集子集。这个方法使得数据集的操作和存档只需要维护一份文本文件，对该文本文件做版本管理即可。要使用时再通过该文件合并出需要的数据集。

**参数：**

- **fileName:** string
  保存添加列表的文本文件。其中的列表遵循以下格式：

  ```shell
  bottle/instances_all.json, bottle/all_bottles
  knife/instances_all.json, knife/all_knives
  [标注文件位置], [图片位置]
  ```

  或

  ```shell
  bottle/instances_all.json
  knife/instances_all.json
  [标注文件位置]
  ```

- inplace: bool
  是否立刻调用`save`方法保存。如为`False`则不调用，需要稍后自己调用`save`方法。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()  # 完全使用默认配置新建一个空对象
newSet.addBatch(fileName="训练数据集v2.1.txt", inplace=True)
```

### save

```python
def save(fileName, replace=False)
```

保存当前所有的标注到由`fileName`指定的标注文件。

**参数：**

- **fileName:** string
  输出的标注文件位置。

- **replace:** bool

  若文件已存在，是否覆盖。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()  # 完全使用默认配置新建一个空对象
newSet.addBatch(fileName="训练数据集v2.1.txt", inplace=True)
newSet.save(fileName="instance_all_数据集v2.3.json", replace=True)
```

### mergeFiles

```python
def mergeFiles(replace=False)
```

将当前`annotation`对象中保存的所有数据集的图片合并到一起。输出路径为`self.name+self.version`。

**参数：**

- **replace:** bool
  若文件已存在，是否覆盖（会移除重复的目录并重新写入，不可逆操作）。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()  # 完全使用默认配置新建一个空对象
newSet.addBatch(fileName="训练数据集v2.1.txt", inplace=True)
newSet.mergeFiles(replace=True)
```

### checkAttributeIntegrity

```python
def checkAttributeIntegrity(tagSet = {}):
```

检查当前已经读取的所有标注标签是否完整。默认的标签列表基于开会讨论的内容：

```python
tagSet = {"overlap", "hard", "visibility", "background_complex", "outdoor", "blur", "small_size", "over_crowded"}
```

**参数：**

- **tagSet:** set
  集合形式的标签列表。使用默认值即可，不用额外传入。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()
newSet.addBatch(fileName="训练数据集v2.1.txt", inplace=True)
newSet.checkAttributeIntegrity()
```

### display

```python
def display(displayPath = "display")
```

简单的输出工具，遍历数据集中所有标注，将预测框、实例分割、标签等信息打印在图片上，输出到`displayPath`。此函数从默认的位置读取图片，所以如果检测到当前对象上有未合并的数据集，会自动运行`mergeFiles(replace=False)`。

**参数：**

- **displayPath:** string
  输出的文件夹名。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()
newSet.addBatch(fileName="训练数据集v2.1.txt", inplace=True)
newSet.display(displayPath="display")
```

###split 

*此函数仅为临时使用，待进一步开发。*

```python
def split(fileName, train=5, test=1, val=1)
```

按照`train:test:val`的比例，把当前数据集拆分成训练集、测试集和验证集。由于是随机拆分，这种方式无法兼顾各种类型图片的兼顾。之后需要另外写一个函数另外处理拆分。

拆分后，会在当前工作目录下生成6个文件：

```shell
工作目录
├── train.txt
├── test.txt
├── val.txt
├── train.json
├── test.json
└── val.json
```

**参数：**

- fileName: string
  输出的标注文件文件名前缀。
- train: int
  训练数据的占比。
- test: int
  测试数据的占比。
- val: int
  验证数据的占比。

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()
newSet.addBatch(fileName="训练数据集v2.1.txt", inplace=True)
newSet.split(train=15, test=1, val=1)
```

### generateAttributes

```python
def generateAttributes()
```

对`self.annotations`生成`blur`、`small_size`、`over_crowded`标签。三个方法的具体实现见下。

**参数：**

- 无

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()
newSet.addBatch(fileName="训练数据集v2.1.txt", inplace=True)
newSet.generateAttributes()
newSet.save(replace=False) # 注意generateAttributes()仅对内存中的标注进行了更改，还需要调用save()来保存。
```

### cutoutAugmentation

```python
def cutoutAugmentation()
```

对当前的标注执行cutout增强。建议只对训练数据做增强。

**参数：**

- 无

**使用示例：**

仅支持读取单个标注进行增强：

```python
import DatasetTools as dt
newSet = dt.annotation()
newSet.addSubset(annotPath="dataSetv2_train.json", imagePath="dataset_images")
newSet.cutoutAugmentation()
```

### summary

```python
def summary()
```

生成当前标注的汇总报告，并在`stdout`打印。

**参数：**

- 无

**使用示例：**

```python
import DatasetTools as dt
newSet = dt.annotation()
newSet.addSubset(annotPath="newDataSet_v2-0.json", imagePath="newDataSet_v2-0")
newSet.generateAttributes()
newSet.summary()
```

→

```
+----------------------+----------+----------+
|         源文件        |  图片数量 |  标注数量  |
+----------------------+----------+----------+
| newDataSet_v2-0.json |   1541   |   2491   |
+----------------------+----------+----------+
+------+------+-------+
| blur | True | False |
+------+------+-------+
|      | 144  |  2347 |
+------+------+-------+
+------------+------+-------+
| size_small | True | False |
+------------+------+-------+
|            |  15  |  2476 |
+------------+------+-------+
+--------------+------+-------+
| over_crowded | True | False |
+--------------+------+-------+
|              | 806  |  1685 |
+--------------+------+-------+
```

### interplSeg

```python
@staticmethod
def interplSeg(segmentation, minDist=5)
```

静态方法，可无需类实例直接调用。输入实例分割+最小点距离，返回依据最小点距离进行过插值的实例分割。

**参数：**

- **segmentation:** list
  实例分割的点集。格式：`[[x1, y1], [x2, y2], ...[xn, yn]]`
- **minDist:** int
  两点间最小距离，若小于此距离就会触发插值。

**返回值：**

- **newSeg:** list
  经插值处理的实例分割点集。格式：`[[x1, y1], [x2, y2], ...[xn, yn]]`

使用示例：

```python
import DatasetTools as dt
seg = ...
newSeg = dt.annotation.interplSeg(seg, minDist=10)
```

## 开发中的功能

- [ ]  根据标签执行更好的数据集分割
- [ ]  不同数据增强方法的植入（也可能直接植入训练pipeline）
- [ ]  ...

