import cv2 as cv

def predictAgeAndGender(frame):

    # 加载预训练好的模型
    ageProto = "D:/wodedaima/python/application of face recognition/age_deploy.prototxt"
    ageModel = "D:/wodedaima/python/application of face recognition/age_net.caffemodel"

    genderProto = "D:/wodedaima/python/application of face recognition/gender_deploy.prototxt"
    genderModel = "D:/wodedaima/python/application of face recognition/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # 数据分类标签
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # 加载预训练好的模型
    ageNet = cv.dnn.readNet(ageModel, ageProto)  # 年龄预测模型
    genderNet = cv.dnn.readNet(genderModel, genderProto)  # 性别预测模型
    # 年龄，性别
    age = []
    gender = []

    blob = cv.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    # 性别预测
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender.append(genderList[genderPreds[0].argmax()])
    # 年龄预测
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age.append(ageList[agePreds[0].argmax()])
    # 若一张照片有多张脸，取出所占范围最大的脸的性别和年龄预测
    gender_single = gender[0]
    age_single = age[0]
    # 提取年龄范围
    ss = age_single.split('-')
    s1 = ss[0].split('(')
    s2 = ss[1].split(')')
    age_range = [int(s1[1]), int(s2[0])]
    return gender_single, age_range


if __name__ == '__main__':
    frame = cv.imread('data/Yudaquan/1.jpg')
    age, gender = predictAgeAndGender(frame)
    print(age, gender)