from flask import redirect, url_for, render_template
import os
import cv2
import time
from flask import Flask
from flask import request
from faceRegnigtionModel import Model, DataSet
from cameraDemo import Camera_reader, takeAPicture
from getCameraPics import cameraAutoForPictures
from dataHelper import readonePicSaveFace
import dlib

app = Flask(__name__)


# 读取训练集
def read_name_list(path):
    name_list: list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


def detectOnePicture(path):
    """
    单图识别
    """
    model = Model()
    model.load()
    img = cv2.imread(path)
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    res = u"Sorry, we can not recognize your identification！\n Please register an account later"
    # 在图像上绘制检测到的人脸
    if dets:
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
            img = img[x1:y1, x2:y2]
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            picType, prob = model.predict(img)
            if picType != -1:
                name_list = read_name_list('dataset/')
                name = name_list[picType]
            else:
                name = 'Unregistered'
    else:
        prob = 0
        name = 'Unregistered'
    return res, prob, name


# 主页面
@app.route("/")
def init():
    return render_template('index.html', title=' home')


# 在web上进行在线人脸识别
@app.route("/recognition/")
def recognition():
    camera = Camera_reader()
    camera.build_camera()
    return render_template("index.html", tile="home")


# 用户填写姓名，收集人脸照片
@app.route("/info", methods=['POST', 'GET'])
def info():
    if request.method == "POST":
        """
        当用户发起post请求，读取用户上传的姓名，跳转到save视图函数(开启摄像头，保存用户上传的照片)
        """
        user_name = request.values.get("user_name")
        return redirect(url_for("save", user_name=user_name))

    else:
        """
        显示info(用户填写信息的界面)渲染界面
        """
        return render_template("info.html")


@app.route("/save/<user_name>")
def save(user_name):
    """
    开启摄像头，保存用户上传的照片
    """
    path = 'data/' + str(user_name)
    cameraAutoForPictures(path)
    return render_template("save.html", name=user_name)


@app.route("/processing/<path>")
def processing(path):
    """
    当用户点击处理图片，将会开始预处理图片，将处理好的图片放入dataset/,然后开始更新模型，当完成后返回主页
    """
    path = 'data/' + str(path)
    readonePicSaveFace(path, ".jpg", ".JPG", "png", "PNG", "tiff")
    dataset = DataSet("dataset/")
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()
    return render_template("index.html", tile="home")


@app.route("/login", methods=['POST', 'GET'])
def login():
    if request.method == "POST":
        """
        当用户发起post请求，读取用户上传的姓名，打开摄像头上传照片，跳转到upload
        """
        user_name = request.values.get("user_name")
        return redirect(url_for("upload", user_name=user_name))

    else:
        """
        显示info(用户填写信息的界面)渲染界面
        """
        return render_template("login.html")


@app.route('/upload/<user_name>')
def upload(user_name):
    """
    打开摄像头，抓拍一张照片，进行识别
    :param user_name:
    :return:
    """
    gender, age = takeAPicture(str(user_name))
    return redirect(url_for('show', name=user_name, user_gender=gender, age_low=age[0],age_high=age[1]))


@app.route('/photo')
def show():
    name = request.args.get('name')
    gender = request.args.get('user_gender')
    age_low = request.args.get('age_low')
    age_high = request.args.get('age_high')
    if name is None:
        print("出错了")
    path = 'static/PhotosUpload/' + name + '.jpg'
    start_time = time.time()
    res, prob, name_predict = detectOnePicture(path)
    end_time = time.time()
    execute_time = str(round(end_time - start_time))
    print(name_predict, [age_low,age_high], gender)
    if prob > 0.9 and name == name_predict and int(age_low) >= 15:
        # 识别概率大于0.9并且登陆时的名字与人脸识别预测的名字一致时，登录成功
        res = "Welcome " + name
        age_info = 'between'+str(age_low)+'-'+str(age_high)
        time_info = str(execute_time)+' seconds'
        return render_template('show.html', url=path, user_name=name, information=res, time=time_info, age=age_info, user_gender=gender)
    elif int(age_low) < 15 and prob > 0.9 and name == name_predict:
        return render_template('young.html')
    else:
        # 登陆失败
        return render_template('error.html', information=res)


if __name__ == "__main__":
    print("faceRegnitionDemo")
    app.run(debug=True)
