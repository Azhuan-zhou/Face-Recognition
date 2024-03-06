import os
import cv2
from faceRegnigtionModel import Model
import dlib
from genderAndAge import predictAgeAndGender


threshold = 0.9


# 在验证窗口显示提示信息
def drew_note1(img,show_name, prob, x1, x2, y1, y2, gender, age):
    cv2.putText(img, "press Q/Esc to exit window", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    cv2.putText(img, show_name, (x2, x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(img, str(prob), (x2+5, x1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
    cv2.putText(img, gender, (y2-60, x1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(img, 'age:'+str(age[0])+"-"+str(age[1]), (y2-90, x1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)


# 在识别窗口中添加提示文字
def drew_note2(img, name, x1, x2, y1, y2, count):
    """
    img: 图像
    name: 当前摄像头的人物
    x1,x2,y1,y2: （x2,x1）是识别出人脸框的左上角，（y2,y1)是识别出人脸框的右下角
    """
    cv2.putText(img, "p: take a photo", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(img, (160, 60), (480, 420), (255, 0, 0), 1)
    cv2.putText(img, "Please put your face in this window", (160, 50), cv2.FONT_HERSHEY_SIMPLEX
                , 0.6, (255, 0, 0), 2)
    cv2.putText(img, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX
                , 1.5, (0, 0, 255), 2)
    cv2.putText(img, name, (x2, x1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 1)


def read_name_list(path):
    """
    读取训练集
    """
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class Camera_reader(object):
    def __init__(self):
        self.model = Model()  # 实例化模型
        self.model.load()  # 加载训练后的模型数据
        self.img_size = 128

    def build_camera(self):
        """
        调用摄像头来实时人脸识别
        """
        # 实例化dlib人脸识别器
        detector = dlib.get_frontal_face_detector()
        # 读取训练集中的人物名称列表
        name_list = read_name_list("dataset/")
        cameraCapture = cv2.VideoCapture(0)
        success, frame = cameraCapture.read()
        while success and cv2.waitKey(1) == -1:
            success, frame = cameraCapture.read()
            frame = cv2.flip(frame, 1, dst=None)  # 水平翻转操作的图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 1)
            # 在图像上绘制检测到的人脸
            if dets:  # 如果识别到人脸
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
                    ROI = gray[x1:y1, x2:y2]  # 切片，选出人脸区域
                    ROI = cv2.resize(
                        ROI,
                        (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LINEAR
                    )
                    # 将ROI转换成128*128像素，利用双线性插值法
                    label, prob = self.model.predict(ROI)
                    if prob > threshold:
                        show_name = name_list[label]
                    else:
                        show_name = "Unregistered"

                    """
                    putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
                    图片
                    要添加的文字
                    文字添加到图片上的位置
                    字体的类型
                    字体大小
                    字体颜色
                    字体粗细
                    """
                    ROI = cv2.cvtColor(ROI,cv2.COLOR_GRAY2BGR)
                    gender, age = predictAgeAndGender(ROI)
                    frame = cv2.rectangle(frame, (x2, x1), (y2, y1), (255, 0, 0), 2)
                    drew_note1(frame, show_name, prob, x1, x2, y1, y2, gender, age)
                    cv2.imshow("Recognition", frame)
                # 绘制矩形框标记人像
            else:  # 没有识别到人脸
                cv2.imshow("Recognition", frame)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 按下q或者esc键退出循环
                break
        cameraCapture.release()
        cv2.destroyAllWindows()


def takeAPicture(name):
    saveDir = 'static/PhotosUpload'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)  # 递归创建文件夹目录
    count = 0  # 记录拍照的次数
    # 实例化人脸分类器
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)  # 调用电脑的摄像头拍照，返回一个videocapture的类
    if not cap.isOpened():
        print("摄像头未打开")
        exit(0)
    # 设置电脑摄像头的参数
    Height = 480
    Width = 680
    w = 360
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
    # Height是镜头高度， Width是镜头宽度, w是后续对图像切片后的宽度和高度
    crop_w_start = (Width - w) // 2  # 160
    crop_h_start = (Height - w) // 2  # 60
    # (crop_w_start , crop_h_start)是摆放位置的左上角
    while True:
        # 获取相框
        ret, frame = cap.read()  # 读取一帧的图像，返回一个bool值和图像的数据
        frame = cv2.flip(frame, 1, dst=None)  # 水平翻转操作的图像
        # 前置摄像头获取的画面是非镜面的，即左手会出现在画面的右侧，此处使用flip进行水平镜像处理
        dets = detector(frame, 1)  # 识别frame图像中人脸的位置，返回top,bottom,left,right
        frame_slice = frame[crop_h_start:crop_h_start + w, crop_w_start:crop_w_start + w]  # 将图像切片成矩形，方便后续模型的构建
        action = cv2.waitKey(1) & 0xFF
        if dets:
            if action == ord('p'):
                count += 1
                cv2.imwrite("%s/%s.jpg" % (saveDir, str(name)),
                            cv2.resize(frame_slice, (128, 128), interpolation=cv2.INTER_AREA)
                            )
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
                ROI = frame[x1:y1, x2:y2]  # 切片，选出人脸区域
                ROI = cv2.resize(
                    ROI,
                    (128, 128),
                    interpolation=cv2.INTER_LINEAR
                )
                drew_note2(frame, name, x1, x2, y1, y2, count)
                cv2.imshow("Uploading", frame)
            gender, age = predictAgeAndGender(ROI)

        else:
            cv2.imshow('Uploading', frame)
        cv2.waitKey(1)
        if count == 1:
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 丢弃窗口
    return gender, age


if __name__ == "__main__":
    camera = Camera_reader()
    camera.build_camera()
