import os
import cv2
import dlib


# 在frame中添加提示文字
def drew_note(img, name, x1, x2, y1, y2, count):
    """
    img: 图像
    name: 当前摄像头的人物
    x1,x2,y1,y2: （x2,x1）是识别出人脸框的左上角，（y2,y1)是识别出人脸框的右下角
    """
    cv2.putText(img, "Q/Esc: exit window", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "c: change your file direction", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "p: take a photo", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(img, (160, 60), (480, 420), (255, 0, 0), 1)
    cv2.putText(img, "Please put your face in this window", (160, 50), cv2.FONT_HERSHEY_SIMPLEX
                , 0.6, (255, 0, 0), 2)
    cv2.putText(img, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX
                , 1.5, (0, 0, 255), 2)
    cv2.putText(img, name, (x2, x1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 1)


def cameraAutoForPictures(saveDir='data/'):  # 传入一个路径
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)  # 递归创建文件夹目录
    count = 1  # 记录拍照的次数
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
    print(f"摄像头的参数(高：{Height},宽{Width})")
    # 设置人脸摆放位置
    crop_w_start = (Width - w) // 2  # 160
    crop_h_start = (Height - w) // 2  # 60
    # (crop_w_start , crop_h_start)是摆放位置的左上角
    detector = dlib.get_frontal_face_detector()  # 实例化dlib正脸人脸识别器
    while True:
        # 获取相框
        ret, frame = cap.read()  # 读取一帧的图像，返回一个bool值和图像的数据
        frame = cv2.flip(frame, 1, dst=None)  # 水平翻转操作的图像
        # 前置摄像头获取的画面是非镜面的，即左手会出现在画面的右侧，此处使用flip进行水平镜像处理
        cv2.imshow("capture", frame)  # 在窗口显示画面
        dets = detector(frame, 1)  # 识别frame图像中人脸的位置，返回top,bottom,left,right
        frame_slice = frame[crop_h_start:crop_h_start + w, crop_w_start:crop_w_start + w]  # 将图像切片成矩形，方便后续模型的构建
        action = cv2.waitKey(1) & 0xFF
        if dets:
            # 当识别到人脸时会在窗口中显示出提示信息，包括姓名，放置人脸位置窗口，提示信息
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
                name = os.path.basename(saveDir)  # 从路径中将姓名提取出来
                drew_note(frame, name, x1, x2, y1, y2, count - 1)
                cv2.imshow("capture", frame)
            if action == ord('c'):
                saveDir = input(u"请输入新的存储目录：")
                saveDir = os.path.join("data/", saveDir)
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
            elif action == ord('p'):
                cv2.imwrite("%s/%d.jpg" % (saveDir, count),
                            cv2.resize(frame_slice, (128, 128), interpolation=cv2.INTER_AREA))
                print(u"%s: %d张图片" % (saveDir, count))
                count += 1
            if action == ord('q') or action == 27:
                break
        else:
            cv2.imshow("capture", frame)
        cv2.waitKey(1)
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 丢弃窗口


if __name__ == '__main__':
    # Azhuan为保存照片的文件名
    cameraAutoForPictures(saveDir='data/Azhuan')
