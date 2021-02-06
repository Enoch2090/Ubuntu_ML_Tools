import json
import os
import shutil
import time
import random
import copy
import cv2
import math
import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot as plt

__version__ = "2.0.0"


class annotation(dict):
    '''
    将多个.json标注文件合并为同一个。导入时注意原有标注需要遵守以下格式，以免发生预料之外的错误：
        - 标注遵守COCO格式，包含一些额外的标签信息。
        - 各个数据集的图片分别存放在对应的文件夹里，建议图片的命名采用"数据集名称+uuid+.jpg"，保证混合后没有重名现象。
    使用方法：
        1. a = annotation() 新建一个对象
        2. 单独使用 a.addSubset() 添加子数据集
        3. 完成添加后用 a.save() 保存合并的标注文件，用 a.mergeFiles() 合并图片。
        4. 使用 a.addBatch() 批量添加。Batch存放在文本文件中，从 fileName 参数传入文件名。文本格式：
            instances_knife_2.json, knife_2
            instances_firearm_1.json, firearm_1
            [PATH TO ANNOTATION FILE], [DIRECTORY OF IMAGES]
    '''

    def __init__(self, name="SentryDataSet", version: str = "2.0", url: str = "", annotPath: str = "", imgPath: str = ""):
        '''
        Args:
            - annotPath: 初始数据集标注文件的位置。
            - imgPath: 存放初始数据集图片的文件夹位置。
            - version: 本次合并后输出的数据集版本号。
            - url: 本次合并后数据集的说明页面，建议填写Notion页面的链接。
            所有参数均为可选，其中annotPath和imgPath两项建议不填，这样类的初始化是一个空数据集，之后再逐步添加。
        '''
        try:
            with open(annotPath, "r") as f:
                ALL = json.load(f)
                self.images = ALL["images"]
                self.annotations = ALL["annotations"]
                self.categories = ALL["categories"]
                self.licenses = ALL["licenses"]
                self.info = ALL["info"]
        except:
            if annotPath != "":
                print("文件%s未找到。" % annotPath)
            print("使用默认配置创建新的总标注对象%s v%s。" % (name, version))
            self.images = []
            self.annotations = []
            self.categories = []
            self.licenses = [{"name": "", "id": 0, "url": ""}]
            self.info = {
                "description": "地面哨兵数据集v%s" % version,
                "url": url,
                "version": version,
                "year": int(time.strftime("%Y", time.localtime())),
                "contributor": "上海交通大学",
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
        self.dirList = [] if imgPath == "" else [imgPath]
        self.imageId = len(self.images)
        self.annotId = len(self.annotations)
        self.categoryId = len(self.categories)
        self.categoryDict = {x["name"]: x["id"] for x in self.categories}
        self.version = version
        self.name = name
        self.table = PrettyTable(["源文件", "图片数量", "标注数量"])
        self.attrTables = []
        self.status = {"merged": True, "subsetNum": 0}
        self.latestImgP = ""
        self.latestAnnot = ""

    def _incrImageId(self):
        self.imageId += 1
        return self.imageId

    def _incrAnnotId(self):
        self.annotId += 1
        return self.annotId

    def _incrCategoryId(self):
        self.categoryId += 1
        return self.categoryId

    @staticmethod
    def dist(this, that):
        return math.sqrt((this[0]-that[0])**2 + (this[1]-that[1])**2)

    @staticmethod
    def interplSeg(segmentation, minDist=5):
        lastPoint = segmentation[-1]
        newSeg = []
        for point in segmentation:
            if annotation.dist(point, lastPoint) <= minDist:
                continue
            lastInserted = lastPoint
            interplSeg = []
            while annotation.dist(point, lastInserted) > minDist:
                lastInserted = [
                    lastInserted[0] + minDist / (annotation.dist(
                        point, lastInserted)) * (point[0]-lastInserted[0]),
                    lastInserted[1] + minDist /
                    (annotation.dist(point, lastInserted)) *
                    (point[1]-lastInserted[1])
                ]
                interplSeg.append(lastInserted)
            newSeg += interplSeg
            # TODO: 还未测试效果
        return newSeg

    def addSubset(self, annotPath="instances_all.json", imagePath=""):
        currAnnot = []
        imageTranslation = {}
        categoryTranslation = {}
        translatedImages = []
        translatedAnnots = []
        imageCounter = 0
        annotCounter = 0
        self.latestAnnot = annotPath
        self.latestImgP = imagePath
        with open(annotPath, "r") as f:
            currAnnot = json.load(f)
        # TODO: 合并类别
        if imagePath != "":
            self.dirList.append(imagePath)
        for category in currAnnot["categories"]:
            if (category["name"] in self.categoryDict.keys()):
                categoryTranslation[category["id"]
                                    ] = self.categoryDict[category["name"]]
                continue
            newCategoryId = self._incrCategoryId()
            self.categoryDict[category["name"]] = newCategoryId
            self.categories.append({
                "id": newCategoryId,
                "name": category["name"],
                "supercategory": category["supercategory"]
            })
            categoryTranslation[category["id"]] = newCategoryId
        for image in currAnnot["images"]:
            newImageId = self._incrImageId()
            imageTranslation[image["id"]] = newImageId
            image["id"] = newImageId
            translatedImages.append(image)
            imageCounter += 1
        for annotation in currAnnot["annotations"]:
            annotation["id"] = self._incrAnnotId()
            annotation["image_id"] = imageTranslation[annotation["image_id"]]
            annotation["category_id"] = categoryTranslation[annotation["category_id"]]
            translatedAnnots.append(annotation)
            annotCounter += 1
        self.images += translatedImages
        self.annotations += translatedAnnots
        self.table.add_row([annotPath, imageCounter, annotCounter])
        self.status["merged"] = True and (self.status["subsetNum"] <= 1)

    def save(self, fileName: str, replace: bool = False) -> bool:
        if os.path.exists(fileName) and not replace:
            fileName = fileName.replace(".json", "-1.json")
            print("标注文件已存在，改为导出到%s。" % fileName)
        data = {
            "licenses": self.licenses,
            "info": self.info,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations
        }
        with open(fileName, "w") as f:
            json.dump(data, f)
            print("标注已导出到 %s。" % (fileName))
        print("统计：")
        self.table.add_row(["总量", self.imageId, self.annotId])
        self.split(fileName=fileName.replace(".json", ""))
        self.summary()
        return True

    def summary(self):
        print(self.table)
        for t in self.attrTables:
            print(t)

    def mergeFiles(self, replace: bool = False) -> bool:
        if len(self.dirList) == 0:
            return False
        outPath = "%sv%s" % (self.name, self.version.replace(".", "-"))
        if os.path.exists(outPath) and not replace:
            print("目标路径 %s 已存在，中止拷贝过程。" % outPath)
            return False
        elif os.path.exists(outPath) and replace:
            print("目标路径 %s 已移除，开始拷贝。" % outPath)
            shutil.rmtree(outPath)
        os.mkdir(outPath)
        try:
            for directory in self.dirList:
                for image in os.listdir(directory):
                    src = os.path.join(directory, image)
                    dst = os.path.join(outPath, image)
                    shutil.copyfile(src, dst)
                    print("%s → %s" % (src, dst))
        except Exception as e:
            print(e)
            return False
        self.status["merged"] = True
        self.dirList = [outPath]
        return True

    def addBatch(self, fileName, inplace: bool = False) -> bool:
        annotList = []
        dirList = []
        try:
            with open(fileName, "r") as f:
                allList = f.readlines()
                annotList = [x.replace("\n", "").replace(
                    " ", "").split(",")[0] for x in allList]
                dirList = []
                try:
                    dirList = [x.replace("\n", "").replace(
                        " ", "").split(",")[1] for x in allList]
                except:
                    pass  # 传入列表不包含图片路径
        except Exception as e:
            print(e)
            return False
        for i in range(len(annotList)):
            self.addSubset(annotPath=annotList[i], imagePath="" if len(
                dirList) == 0 else dirList[i])
        if inplace:
            return self.save(fileName="%s_v%s.json" % (self.name, self.version.replace(".", "-")), replace=True) and self.mergeFiles(replace=True)
        return False

    def split(self, fileName: str, train: int = 5, test: int = 1, val: int = 1):
        print("按 train:test:val = %s:%s:%s 创建分割……" % (train, test, val))
        trainPercentage = train / (train + test + val)
        testPercentage = test / (train + test + val)
        trainLen = int(len(self.images) * trainPercentage)
        testLen = int(len(self.images) * testPercentage)
        allImageList = [x["file_name"] for x in self.images]
        self.trainList = list(random.sample(set(allImageList), trainLen))
        self.testList = list(random.sample(
            set(allImageList) - set(self.trainList), testLen))
        self.valList = list(set(allImageList) -
                            set(self.trainList) - set(self.testList))
        table = PrettyTable(["名称", "数量"])
        table.add_row(["训练集 train", trainLen])
        table.add_row(["测试集 test", testLen])
        table.add_row(["验证集 val", len(self.images) - trainLen - testLen])
        print(table)
        self.trainImages = []
        self.testImages = []
        self.valImages = []
        self.trainAnnots = []
        self.testAnnots = []
        self.valAnnots = []
        print("分割创建完成。正在保存标注文件……")

        image_id = 1
        anno_id = 1
        translation = {}
        for image in self.images:
            image_ = copy.deepcopy(image)
            if (image_["file_name"] in self.trainList):
                translation[image_["id"]] = image_id
                image_["id"] = image_id
                self.trainImages.append(image_)
                image_id += 1
        for annotation in self.annotations:
            annotation_ = copy.deepcopy(annotation)
            if (self.images[annotation_["image_id"]-1]["file_name"] in self.trainList):
                annotation_["image_id"] = translation[annotation["image_id"]]
                annotation_["id"] = anno_id
                self.trainAnnots.append(annotation_)
                anno_id += 1
        trainData = {
            "licenses": self.licenses,
            "info": self.info,
            "categories": self.categories,
            "images": self.trainImages,
            "annotations": self.trainAnnots
        }
        with open("%s_train.json" % fileName, "w") as f:
            json.dump(trainData, f)
            print("训练数据标注保存在 %s。" % ("%s_train.json" % fileName))

        image_id = 1
        anno_id = 1
        translation = {}
        for image in self.images:
            image_ = copy.deepcopy(image)
            if (image_["file_name"] in self.testList):
                translation[image_["id"]] = image_id
                image_["id"] = image_id
                self.testImages.append(image_)
                image_id += 1
        for annotation in self.annotations:
            annotation_ = copy.deepcopy(annotation)
            if (self.images[annotation_["image_id"]-1]["file_name"] in self.testList):
                annotation_["image_id"] = translation[annotation["image_id"]]
                annotation_["id"] = anno_id
                self.testAnnots.append(annotation_)
                anno_id += 1
        testData = {
            "licenses": self.licenses,
            "info": self.info,
            "categories": self.categories,
            "images": self.testImages,
            "annotations": self.testAnnots
        }
        with open("%s_test.json" % fileName, "w") as f:
            json.dump(testData, f)
            print("测试数据标注保存在 %s。" % ("%s_test.json" % fileName))

        image_id = 1
        anno_id = 1
        translation = {}
        for image in self.images:
            image_ = copy.deepcopy(image)
            if (image_["file_name"] in self.valList):
                translation[image_["id"]] = image_id
                image_["id"] = image_id
                self.valImages.append(image_)
                image_id += 1
        for annotation in self.annotations:
            annotation_ = copy.deepcopy(annotation)
            if (self.images[annotation_["image_id"]-1]["file_name"] in self.valList):
                annotation_["image_id"] = translation[annotation["image_id"]]
                annotation_["id"] = anno_id
                self.valAnnots.append(annotation_)
                anno_id += 1
        valData = {
            "licenses": self.licenses,
            "info": self.info,
            "categories": self.categories,
            "images": self.valImages,
            "annotations": self.valAnnots
        }
        with open("%s_val.json" % fileName, "w") as f:
            json.dump(valData, f)
            print("验证数据标注保存在 %s。" % ("%s_val.json" % fileName))
        print("正在保存分割文件……")
        with open("%s_train.txt" % fileName, "w") as f:
            f.writelines([x+"\n" if x != self.trainList[-1]
                          else x for x in self.trainList])
            print("训练分割文件保存在 %s。" % ("%s_train.txt" % fileName))
        with open("%s_test.txt" % fileName, "w") as f:
            f.writelines([x+"\n" if x != self.testList[-1]
                          else x for x in self.testList])
            print("测试分割文件保存在 %s。" % ("%s_test.txt" % fileName))
        with open("%s_val.txt" % fileName, "w") as f:
            f.writelines([x+"\n" if x != self.valList[-1]
                          else x for x in self.valList])
            print("验证分割文件保存在 %s。" % ("%s_val.txt" % fileName))

    def checkAttributeIntegrity(self, tagSet={"overlap", "hard", "visibility", "background_complex", "outdoor", "blur", "small_size", "over_crowded"}
                                ):
        print("正在检查标注的标签是否完整……")
        stdAttributes = tagSet
        wrongAttributeList = {}
        annotId = 1
        for annotation in self.annotations:
            attributes = set(annotation["attributes"].keys())
            if attributes == stdAttributes:
                continue
            lostAttributes = stdAttributes - attributes
            wrongAttributeList[annotId] = lostAttributes
            for attribute in lostAttributes:
                annotation["attributes"][attribute] = "N/A"
        if len(wrongAttributeList) != 0:
            print('使用"N/A"填充了所有缺失标签。')
        else:
            print("检查完成，无标签缺失。")
        for k, v in wrongAttributeList:
            print("编号%s的标注缺失标签：%s" % (k, v))

    def display(self, displayPath="display"):
        self.checkAttributeIntegrity()
        OPT_PATH = displayPath
        IMG_PATH = self.latestImgP
        # 此函数的主代码从display.ipynb迁移而来
        if not(self.status["merged"]):
            self.mergeFiles(replace=False)
        ALL = {}
        IMAGES = self.images
        if os.path.exists(OPT_PATH):
            shutil.rmtree(OPT_PATH)
        os.mkdir(OPT_PATH)
        IMAGES_PATH = {x["id"]: x["file_name"] for x in IMAGES}
        for annotation in self.annotations:
            imgPath1 = os.path.join(
                IMG_PATH, IMAGES_PATH[annotation["image_id"]])
            imgPath2 = os.path.join(
                OPT_PATH, IMAGES_PATH[annotation["image_id"]])
            imgPath = imgPath2 if os.path.exists(imgPath2) else imgPath1
            img = cv2.imread(imgPath)
            img = img.astype(np.int32)
            b, g, r = cv2.split(img)
            it = iter(annotation["segmentation"][0])
            seg = []
            while True:
                try:
                    x = next(it)
                    y = next(it)
                    seg.append([x, y])
                except StopIteration:
                    break
            seg_np = np.array([seg], dtype=np.int32)
            im = np.zeros(img.shape[:2], dtype=np.int32)
            cv2.polylines(im, seg_np, 1, 255)
            cv2.fillPoly(im, seg_np, 255)
            b_masked = cv2.addWeighted(b, 1, im, 0.5, 0)
            g_masked = cv2.addWeighted(g, 1, im, 0.5, 0)
            r_masked = cv2.addWeighted(r, 1, im, 0.5, 0)
            img = cv2.merge([b_masked, g_masked, r_masked])
            [x, y, w, h] = list(map(int, annotation["bbox"]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            text = "Category: %s\n blur: %s\n smallSize: %s\n overCrowded: %s\n overlap: %s\n hard: %s\n visibility: %s\n bkg complex: %s\n outdoor: %s" % (
                self.categories[annotation["category_id"]-1]["name"],
                annotation["attributes"]["blur"],
                annotation["attributes"]["small_size"],
                annotation["attributes"]["over_crowded"],
                annotation["attributes"]["overlap"],
                annotation["attributes"]["hard"],
                annotation["attributes"]["visibility"],
                annotation["attributes"]["background_complex"],
                annotation["attributes"]["outdoor"]
            )
            lineHeight = 15
            count = 0
            for textSeg in text.split("\n"):
                y_ = (y + count*lineHeight) if (y + count *
                                                lineHeight) < b.shape[0] else b.shape[0]
                cv2.putText(img, textSeg, (x, y_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                count += 1
            cv2.imwrite(imgPath2, img)
    #print("Annotation %s -> Image %s" % (annotation["id"], annotation["image_id"]))

    def generateAttributes(self):
        self.generateBlurAttribute()
        self.generateSizeAttribute()
        self.generateOvercrowdedAttribute()

    def generateBlurAttribute(self, showSummary=False):
        #blurdf = pd.DataFrame(columns=["Vollath", "Laplacian", "Entropy", "blurLaplacian"])
        def Vollath(img):
            '''
            :param img:narray 二维灰度图像
            :return: float 图像越清晰越大
            '''
            shape = np.shape(img)
            u = np.mean(img)
            out = -shape[0]*shape[1]*(u**2)
            for x in range(0, shape[0]-1):
                for y in range(0, shape[1]):
                    out += int(img[x, y])*int(img[x+1, y])
            return out/1000

        def entropy(img):
            '''
            :param img:narray 二维灰度图像
            :return: float 图像越清晰越大
            '''
            out = 0
            count = np.shape(img)[0]*np.shape(img)[1]
            p = np.bincount(np.array(img).flatten())
            for i in range(0, len(p)):
                if p[i] != 0:
                    out -= p[i]*math.log(p[i]/count)/count
            return out

        def safe_int(a): return int(a)-1 if int(a)-1 > 0 else int(a)+1
        if not self.status["merged"]:
            print("文件未合并。调用mergeFiles(replace=False)开始拷贝……")
            self.mergeFiles(replace=False)
        outPath = "%sv%s" % (self.name, self.version.replace(".", "-"))
        imageName_ = ""
        image = []
        blurs = []
        trueNum = 0
        falseNum = 0
        if os.path.exists("t"):
            shutil.rmtree("t")
        os.mkdir("t")
        for annotation in self.annotations:
            imageName = self.images[annotation["image_id"]-1]["file_name"]
            if not imageName_ == imageName:  # 减少IO次数
                image = cv2.imread(os.path.join(outPath, imageName))
            imageName_ = imageName
            [x, y, w, h] = list(map(safe_int, annotation["bbox"]))
            imageRoI = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(imageRoI, cv2.CV_64F)
            area = imageRoI.shape[0] * imageRoI.shape[1]
            blurImageRoI = cv2.blur(imageRoI, (3, 3))
            bblurLaplacian = cv2.Laplacian(blurImageRoI, cv2.CV_64F).var()
#             blurVollath = Vollath(imageRoI)/area
            blurLaplacian = cv2.Laplacian(imageRoI, cv2.CV_64F).var()
#             blurEntropy = entropy(imageRoI)
            #blurdf = blurdf.append(pd.DataFrame([[blurVollath, blurLaplacian, blurEntropy, bblurLaplacian]], columns=["Vollath", "Laplacian", "Entropy", "blurLaplacian"]), ignore_index=True)
            blurImageRoI = cv2.resize(blurImageRoI, (w, h))
            if blurLaplacian/bblurLaplacian < 4:
                annotation["attributes"]["blur"] = True
                trueNum += 1
            else:
                annotation["attributes"]["blur"] = False
                falseNum += 1
#             imageOut = np.zeros((5*h,5*w), dtype="uint8")+255
#             imageOut[0:h, 0:w] = imageRoI
#             imageOut[0:h, w+1:2*w+1] = blurImageRoI
#             imageOut = cv2.putText(cv2.cvtColor(imageOut, cv2.COLOR_GRAY2BGR),str(blurLaplacian) , (15, h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             imageOut = cv2.putText(imageOut,str(bblurLaplacian) , (15, h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             imageOut = cv2.putText(imageOut,str(blurLaplacian/bblurLaplacian) , (15, h+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             if blurLaplacian/bblurLaplacian < 4:
#                 cv2.imwrite("t/%s.jpg" % (annotation["id"]), imageOut)
        t = PrettyTable(["blur", "True", "False"])
        t.add_row(["", trueNum, falseNum])
        self.attrTables.append(t)
        # if showSummary:
        #     plt.scatter(blurdf["Laplacian"] /
        #                 blurdf["blurLaplacian"], blurdf["Laplacian"])

    def generateSizeAttribute(self):
        # 实例分割转旋转框by wkl
        def instance2obb(img_size, boundary):
            '''
            Input:
                img_size: image size in format (weight, height)
                boundary: point sets of instance boundary in format (2*n_points, 1) as list; 
                          the same as COCO_JSON["annotations"]["segmentation"]

                Note that only input ONE boundary in one time.

            Output:
                vertexes: point sets of oriented bounding box in format (2*4, 1) as list; 
                          the same as COCO_JSON["annotations"]["segmentation"]
                long_edge: longer edge of obb
                short_edge: shorter edge of obb
                beyond_flag: if true, obb is out of the border of image

            Usage: 
                vertexes, long_edge, short_edge, beyond_flag = instance2obb(img_size, boundary)
                vertexes, long_edge, short_edge, beyond_flag = instance2obb((weight, height), boundary)
            '''

            max_x, max_y = img_size    # max_x = weight; max_y = height

            boundary = np.reshape(boundary, [-1, 2]).astype(np.float32)
            obb = cv2.minAreaRect(boundary)
            long_edge, short_edge = max(obb[1]), min(obb[1])
            vertexes = cv2.boxPoints(obb)

            beyond_flag = False
            for vertex_x, vertex_y in vertexes:    # Note: x->w; y->h
                if (vertex_x < 0) or (vertex_x > max_x) or (vertex_y < 0) or (vertex_y > max_y):
                    beyond_flag = True
                    break

            vertexes = vertexes.astype(np.float64).round(2).flatten().tolist()

            return vertexes, long_edge, short_edge, beyond_flag

        trueNum = 0
        falseNum = 0
        for annotation in self.annotations:
            imageName = self.images[annotation["image_id"]-1]["file_name"]
            imageWidth = self.images[annotation["image_id"]-1]["width"]
            imageHeight = self.images[annotation["image_id"]-1]["height"]
            v, l, s, b = instance2obb(
                (imageWidth, imageHeight), annotation["segmentation"])
            if l <= 24:
                annotation["attributes"]["small_size"] = True
                trueNum += 1
                continue
            annotation["attributes"]["small_size"] = False
            annotation["attributes"]["obb_beyond_boundary"] = b
            falseNum += 1
        t = PrettyTable(["size_small", "True", "False"])
        t.add_row(["", trueNum, falseNum])
        self.attrTables.append(t)

    def generateOvercrowdedAttribute(self, distThreshProportion=0.25):
        annotId = 0
        trueNum = 0
        falseNum = 0

        for annotation in self.annotations:
            upAnnotId = annotation["id"]-2
            downAnnotId = annotation["id"]
            bboxes = []
            while (upAnnotId >= 0) and (self.annotations[upAnnotId]["image_id"] == annotation["image_id"]):
                bboxes.append(self.annotations[upAnnotId]["bbox"])
                upAnnotId -= 1
            while (downAnnotId < len(self.annotations)) and (self.annotations[downAnnotId]["image_id"] == annotation["image_id"]):
                bboxes.append(self.annotations[downAnnotId]["bbox"])
                downAnnotId += 1
            bboxCenters = [[x+w/2, y+h/2] for [x, y, w, h] in bboxes]
           # print(bboxCenters)
            [x, y, w, h] = annotation["bbox"]
            currCenter = [x+w/2, y+h/2]
            width = self.images[annotation["image_id"]-1]["width"]
            height = self.images[annotation["image_id"]-1]["height"]
            distThresh = distThreshProportion * \
                (width if width < height else height)
            isOverCrowded = False
            for bboxCenter in bboxCenters:
                if annotation.dist(currCenter, bboxCenter) <= distThresh:
                    isOverCrowded = True
                    break
            annotation["attributes"]["over_crowded"] = isOverCrowded
            if isOverCrowded:
                trueNum += 1
            else:
                falseNum += 1
        t = PrettyTable(["over_crowded", "True", "False"])
        t.add_row(["", trueNum, falseNum])
        self.attrTables.append(t)

    def cutoutAugmentation(self):
        if len(self.annotations) == 0 or len(self.dirList) == 0:
            print("请先读取一个训练集的标注。中断Cutout增强。")
            return
        if len(self.dirList) > 1:
            print("仅支持单个标注文件的扩充。中断Cutout增强。")
            return
        PTSRATIO = 0.25
        basePath = self.dirList[0]
        targetPath = self.dirList[0] + "_cutoutAugmented"

        class imageList(list):
            def __init__(self, other=[]):
                list.__init__([])
                self.extend(other)

            def getFileName(self, Id):
                return self[Id-1]["file_name"]

            def getImageShape(self, Id):
                return [self[Id-1]["width"], self[Id-1]["height"]]

        def safe_int(a): return int(a)-1 if int(a)-1 > 0 else int(a)+1

        # 此函数的主代码从preprocess.ipynb迁移而来
        if os.path.exists(targetPath):
            shutil.rmtree(targetPath)
        os.mkdir(targetPath)
        ALL = {}
        imageFileList = imageList(self.images)
        for annotation in self.annotations:
            imageFile = imageFileList.getFileName(annotation["image_id"])
            [imageWidth, imageHeight] = imageFileList.getImageShape(
                annotation["image_id"])
            imagePath1 = os.path.join(basePath, imageFile)
            imagePath2 = os.path.join(targetPath, imageFile)
            image = []
            if not os.path.exists(imagePath2):
                image = cv2.imread(imagePath1)
            else:
                image = cv2.imread(imagePath2)
            [b, g, r] = cv2.split(image)
            imageAvrColor = (int(b.mean()), int(g.mean()), int(r.mean()))
            it = iter(annotation["segmentation"][0])
            seg = []
            while True:
                try:
                    x = next(it)
                    y = next(it)
                except StopIteration:
                    seg.append(x)
                    break
                seg.append([safe_int(x), safe_int(y)])
            seg = annotation.interplSeg(seg)
            segSelStart = random.randint(0, int(len(seg)*(1-PTSRATIO))-1)
            segSelEnd = segSelStart + int(len(seg)*PTSRATIO)-1
            segSel = seg[segSelStart:segSelEnd]
            xMax, yMax, xMin, yMin = [1, 1, imageWidth-1, imageHeight-1]
            for [x_, y_] in segSel:
                xMax = x_ if x > xMax else xMax
                yMax = y_ if y > yMax else yMax
                xMin = x_ if x <= xMin else xMin
                yMin = y_ if y <= yMin else yMin
            startPoint = (int(xMin), int(yMin))
            endPoint = (int(xMax), int(yMax))
            # TODO: 改变标注文件
            thickness = -1
            image = cv2.rectangle(
                img=image, pt1=startPoint, pt2=endPoint, color=imageAvrColor, thickness=thickness)
            cv2.imwrite(imagePath2, image)
            print(imagePath2)
