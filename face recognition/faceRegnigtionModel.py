import os
import cv2
import random
import numpy as np
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.utils.np_utils import to_categorical

"""
编写类DataSet,保存和读取格式化后的训练数据
"""


class DataSet(object):
    # 初始化

    def __init__(self, path):
        self.num_classes = None  # 数据集中的人数
        self.X_train = None  # 训练集
        self.X_test = None  # 测试集
        self.Y_train = None  # 训练集标签
        self.Y_test = None  # 测试机标签
        self. img_size = 128  # 训练集中图像的大小128*128*1
        self.extract_data(path)  # 将数据集中的数据分为训练集和测试集

    # 抽取数据
    def extract_data(self, path: str):
        imgs, labels, counter = read_file(path)  # imgs: 所有图片数据， labels: 图片对应的标签， counter: 标签的个数
        X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels, test_size=0.2,
                                                            random_state=random.randint(0, 100))
        X_train = X_train.reshape(X_train.shape[0], 1, self.img_size, self.img_size)/255.0  # 标准化
        X_test = X_test.reshape(X_test.shape[0], 1, self.img_size, self.img_size) / 255.0  # 标准化
        # 转化为浮点
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # 热编码
        Y_train = to_categorical(Y_train, counter)
        Y_test = to_categorical(Y_test, counter)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

    # 数据校验
    def check(self):
        print("num of dim:", self.X_test.ndim)
        print("shape:", self.X_test.shape)
        print("size:", self.X_test.size)
        print("num of dim:", self.X_train.ndim)
        print("shape:", self.X_train.shape)
        print("size:", self.X_train.size)


def endwith(s, *endstring):
    """
    对字符串的后续和标记进行匹配
    :param s: str
    :param endstring: str
    :return: bool
    """
    resultArray: map = map(s.endswith, endstring)
    if True in resultArray:
        return True
    else:
        return False


# 读取指定路径的图片信息
def read_file(path):
    # path是路径‘dataset/’
    img_list = []  # 保存图像信息
    label_list: list = []  # 被划分图像样本的标记
    dir_counter: int = 0  # 记录路径’dataset/’下有多少个文件夹
    IMG_SIZE: int = 128
    for child_dir in os.listdir(path):
        # child_dir是目录’dataset/‘所包含的文件夹名称
        child_path = os.path.join(path, child_dir)
        # 合并成‘dataset/child_dir'，传递给child_path
        for dir_image in os.listdir(child_path):
            # 在文件夹‘dataset/child_dir'下，将所有的文件名称传递给dir_image
            if endwith(dir_image, 'jpg'):
                # 判度文件名称dir_image是否包含jpg后缀
                img = cv2.imread(os.path.join(child_path, dir_image))
                # 如果包含jpg后缀，则为图片。将该图片读取，转化为一个numpy数组
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                #将图片像素转换为128*128
                recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                # 转化成灰度照片
                img_list.append(recolored_img)
                label_list.append(dir_counter)
        dir_counter += 1
    img_list = np.array(img_list)
    return img_list, label_list, dir_counter


# 读取训练集
def read_name_list(path):
    name_list: list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


"""
创建一个基于CNN的人脸识别模型，开始构建数据模型并进行训练
"""


class Model(object):
    FILE_PATH = "face.h5"
    IMAGE_SIZE = 128
    dataset: DataSet = None

    def __init__(self):
        self.model = None

    def read_trainData(self, data_set: DataSet):
        self.dataset = data_set

    def build_model(self):
        self.model = Sequential()
        # 层1，卷积层，输入图片128*128*1，输出图片128*128*32, 修改成32个通道， 3*3 的卷积核
        self.model.add(
            Convolution2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                # dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )
        # 层2：relu激活
        self.model.add(Activation('relu'))
        # 层3: 池化层, 输入图片128*128*32，输出图片64*64*32,池化降维，减小复杂度
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        # 层4： 卷积层，输入图片64*64*32，输出图片64*64*64
        self.model.add(
            Convolution2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same'
            )
        )
        # 层5： relu激活函数
        self.model.add(Activation('relu'))
        # 层6： 池化层，输入图片64*64*64，输出图片32*32*64
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        # 层7： 卷积层，输入图片32*32*64，输出图片32*32*64，修改成64个通道，3*3的卷积核
        self.model.add(
            Convolution2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same'
            )
        )
        # 层8： relu激活函数
        self.model.add(Activation('relu'))
        # 层9： 池化层，输入图片32*32*64，输出图片16*16*64
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        #层10：以0.5的概率让神经元失活，防止过拟合
        self.model.add(Dropout(0.5))
        # 层11： 展平 输入16*16*64，输出1*16384
        self.model.add(Flatten())
        # 层12：全连接，1024个神经元
        self.model.add(Dense(1024))
        # 层13： relu激活函数
        self.model.add(Activation('relu'))
        # 层14：全连接，num_classes个神经元
        self.model.add(Dense(self.dataset.num_classes))
        # 层15：softmax激活函数
        self.model.add(Activation('softmax'))
        # 显示神经网络构架
        self.model.summary()
    # 训练模型
    def train_model(self):
        # 优化器：adam, 损失函数：交叉熵损失函数，评价指标：正确率
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=10, batch_size=10)
        # 12次训练，每一次的训练中，使用50张照片对模型进行一次反向传播参数更新

    # 用测试集，评价模型
    def evaluate_model(self):
        print('\nTesting--------------------')
        loss, accuracy = self.model.evaluate(
            self.dataset.X_test,
            self.dataset.Y_test
        )
        print("test loss：", loss)
        print("test accuracy:", accuracy)

    # 保存训练好的模型
    def save(self, file_path=FILE_PATH):
        print("Model Saved Finished!!!!")
        self.model.save(file_path)

    # 加载训练后的模型
    def load(self, file_path=FILE_PATH):
        print("Mode Loaded Successful!!!!")
        self.model = load_model(file_path)

    def predict(self, img):
        # 更改数组的形状为 1*1*128*128(图片个数，通道个数，图片的像素尺寸）
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')  # 转换将numpy数据整数类型转换为浮点型
        img = img/255.0  # 归一化
        # 返回每个标签的预测结果
        result = self.model.predict(img)
        # 找到结果中概率最大的小标
        max_index = np.argmax(result)
        return max_index, result[0][max_index]


if __name__ == "__main__":
    dataset = DataSet("dataset/")
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()
