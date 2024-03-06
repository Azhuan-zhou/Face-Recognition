import os
import cv2
import time
import dlib
import random


def readAllImg(path, *suffix):
    """
    基于后缀读取文件
    :param path: str
    :param suffix: tuple
    :return: list
    """
    try:
        s: list = os.listdir(path)  # 返回path目录下所有文件名字的列表，以字母顺序
        resultArray: list = []  # 用于存放
        fileName: str = os.path.basename(path)  # 返回path路径中最后一个组成部分
        resultArray.append(fileName)  # 将filename拼接到resultArray
        for i in s:
            if endwith(i, *suffix):  # 判度path路径中的文件名i是否包含后缀suffix
                document: str = os.path.join(path, i)  # 合并路径path与文件名i
                img = cv2.imread(document)  # 读取图片，返回一个numpy数组给img
                resultArray.append(img)  # 将图像数据保存给resultArray列表
    except IOError:
        print("Error")
    else:
        return resultArray


def endwith(s, *endstring):
    """
    对字符串后缀进行匹配
    :param s: str
    :param endstring: tuple
    :return: bool
    """
    resultArray: map = map(s.endswith, endstring)  # 查找字符串s是否有endstring中的后缀
    if True in resultArray:
        return True
    else:
        return False


#  改变图像的亮度，增加数据的多样性，light表示对比度，bias表示亮度偏置
def img_change(img, light, bias):
    width = img.shape[1]   # 图像的水平尺寸（宽度）
    height = img.shape[0]  # 图像的垂直尺寸（高度）
    #  范围
    for i in range(0, width):  # 给i赋值
        for j in range(0, height):
                # 线性调整 light表示对比度，bias代表亮度偏置
                tmp = int(img[j, i] * light + bias)
                if tmp > 255:
                    tmp = 255

                elif tmp < 0:
                    tmp = 0
                img[j, i] = tmp

    return img

# 将data文件夹中的某个名字文件中的所有图片进行转换
def readonePicSaveFace(path, *suffix):
    """
    图片标准化与储存
    # path: 相对路径 data/<name>
    # *suffix: 后缀
    """
    # 判度是否存在目标路径，如果没有则创建一个
    try:
            name = os.path.basename(path)
            print(name)
            objectpath = os.path.join('dataset/', name)
            if not os.path.exists(objectpath):
                os.makedirs(objectpath)
            resultArray: list = readAllImg(path, *suffix)
            # 列表第一个元素是sourcePath中最后一个文件，其余是图像的信息
            count = 1
            # 实例化dlib人脸识别器
            detector = dlib.get_frontal_face_detector()
            for i in resultArray:
                if type(i) != str:  # 保证取到的是图像的数据，因为resultArray列表中第一个元素是str
                    # 将图像转化为灰度图像
                    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                    """--------------------------------------------------------------------
                    使用enumerate 函数遍历序列中的元素以及它们的下标,i为人脸序号,d为i对应的元素;
                    left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离 
                    top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
                    ----------------------------------------------------------------------"""
                    dets = detector(gray, 1)
                    # 在图像上绘制检测到的人脸
                    for j, d in enumerate(dets):
                        x1 = d.top()
                        if x1 >= 0:
                            pass
                        else:
                            break
                        y1 = d.bottom()
                        if y1 >= 0:
                            pass
                        else:
                            break
                        x2 = d.left()
                        if x2 >= 0:
                            pass
                        else:
                            break
                        y2 = d.right()
                        if y2 >= 0:
                            pass
                        else:
                            break
                        face = gray[x1:y1, x2:y2]
                        """调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性"""
                        face = img_change(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                        face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)  # 调整图片的大小
                        listStr = [str(int(time.time())), str(count)]
                        filename = ''.join(listStr)
                        cv2.imwrite(objectpath + os.sep + '%s.jpg' % filename, face)
                        count += 1
            print("Read " + str(count - 1) + " Face to Destination " + objectpath)
    except Exception as e:
        print("Exception:", e)


def readPicSaveFace(path, *suffix):
    """
    将data/目录中所有文件中的图片标准化与储存
    path: 是和该文件同一级的文件夹data/
    """
    # 判度是否存在目标路径，如果没有则创建一个
    try:
        a = os.listdir(path)
        for child_dir in os.listdir(path):
            child_sourcepath = os.path.join(path, child_dir)
            child_objectpath = os.path.join('dataset/', child_dir)
            if not os.path.exists(child_objectpath):
                os.makedirs(child_objectpath)
            resultArray: list = readAllImg(child_sourcepath, *suffix)
            # 列表第一个元素是sourcePath中最后一个文件，其余是图像的信息
            count = 1
            # 加载级联分类器文件
            detector = dlib.get_frontal_face_detector()
            for i in resultArray:
                if type(i) != str:  # 保证取到的是图像的数据，因为resultArray列表中第一个元素是str
                    # 将图像转化为灰度图像
                    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                    """--------------------------------------------------------------------
        使用enumerate 函数遍历序列中的元素以及它们的下标,i为人脸序号,d为i对应的元素;
        left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离 
        top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
        ----------------------------------------------------------------------"""
                    dets = detector(gray, 1)
                    # 在图像上绘制检测到的人脸
                    for j, d in enumerate(dets):
                            x1 = d.top()
                            if x1 >= 0:
                                pass
                            else:
                                break
                            y1 = d.bottom()
                            if y1 >= 0:
                                pass
                            else:
                                break
                            x2 = d.left()
                            if x2>= 0:
                                pass
                            else:
                                break
                            y2 = d.right()
                            if y2>= 0:
                                pass
                            else:
                                break
                            face = gray[x1:y1, x2:y2]
                           # """调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性"""
                            face = img_change(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                            face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)  # 调整图片的大小
                            listStr = [str(int(time.time())), str(count)]
                            filename = ''.join(listStr)
                            cv2.imwrite(child_objectpath + os.sep + '%s.jpg' % filename, face)
                            count += 1
            print("Read " + str(count - 1) + " Face to Destination " + child_objectpath)
    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    print("dataProcessing!!!")
    readonePicSaveFace('data/Yudaquan','.jpg')
    print("Finish!")