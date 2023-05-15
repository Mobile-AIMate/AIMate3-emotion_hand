import cv2                     #  图像处理的库 OpenCv
import dlib                    # 人脸识别的库 dlib
import numpy as np             # 数据处理的库 numpy
import json
import queue
import random
import socket
import threading
import time
import traceback
from datetime import datetime
import mediapipe as mp

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = socket.gethostname()
port = 8888

server_socket.bind((host, port))

server_socket.listen(1)

# 线程安全队列
request_queue = queue.Queue()
data_queue = queue.Queue()

# 定义面部识别
detector = dlib.get_frontal_face_detector()
# dlib 的68点模型，使用官方训练好的特征预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

'''
This .dat file can be downloaded at:
https://pan.baidu.com/s/1oMDX4mzrwqBm9a55zSdERA?pwd=k1bp 
'''

# 定义手，检测对象
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# 直接打开摄像头进行拍摄获取图像并分别进行处理
def create_data():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    i = 0

    while True:

        i += 1

        issuccess, frame = cap.read()
        if not issuccess:
            continue

        if i % 2:
            print(i)
            emotion_data = create_emotion_data(img = frame)
        hand_data = create_hand_data(img = frame)

        data = [emotion_data, hand_data]
        data_queue.put(data)

def create_emotion_data(img):


    # 眉毛直线拟合数据缓冲
    line_brow_x = []
    line_brow_y = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
        # 如果检测到人脸
    if(len(faces) != 0):    
        
        # 对每个人脸都标出68个特征点
        for i in range(len(faces)):
            # enumerate 方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
            for k, d in enumerate(faces):
                # 计算人脸识别框边长
                face_width = d.right() - d.left()
                
                # 使用预测器得到68点数据的坐标
                shape = predictor(img, d)
                    
                # 分析任意 n 点的位置关系来作为表情识别的依据
                # 嘴中心	66，嘴左角48，嘴右角54
                mouth_width = (shape.part(54).x - shape.part(48).x) / face_width #嘴巴张开程度
                mouth_height = (shape.part(66).y - shape.part(62).y) / face_width #嘴巴张开程度
                #print("嘴巴宽度与识别框宽度之比：" , mouth_width)
                #print("嘴巴高度与识别框宽度之比：" , mouth_height)
                
                # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
                brow_sum = 0 #高度之和
                frown_sum = 0  #两边眉毛距离之和
                for j in range(17, 21):
                    brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                    frown_sum += shape.part(j + 5).x - shape.part(j).x
                    line_brow_x.append(shape.part(j).x)
                    line_brow_y.append(shape.part(j).y)
                    
                # self.brow_k, self.brow_d = self.fit_slr(line_brow_x, line_brow_y) # 计算眉毛的倾斜程度
                tempx = np.array(line_brow_x)
                tempy = np.array(line_brow_y)
                z1 = np.polyfit(tempx, tempy, 1)  #拟合成一次直线
                brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的
                
                
                brow_height = (brow_sum / 10) / face_width # 眉毛高度占比
                brow_width = (frown_sum / 5) / face_width  # 眉毛距离占比
                
                #print("眉毛高度与识别框宽度之比：" , brow_height)
                #print("眉毛间距与识别框高度之比：" , brow_width)
                
                # 眼睛睁开程度
                eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y + 
                            shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
                eye_hight = (eye_sum / 4) / face_width
                #print("眼睛睁开距离与识别框高度之比：" , eye_hight)
                
                # 分情况讨论,判断情绪变化
                # 张嘴，可能是开心或惊讶，通过眼睛的睁开程度区分
                if round(mouth_height >= 0.04):
                    if eye_hight >= 0.036:
                        emo_label = "amazing"
                    else:
                        emo_label = "happy"
                        
                # 没有张嘴，可能是正常和生气，通过眉毛区分
                else:
                    if brow_k <= 0.29:
                        emo_label = "sad"
                    else:
                        emo_label = "nature"
                face_position = [int(d.left()), int(d.bottom()), int(d.right()), int(d.top())]
                emotion_data = {'emotion_label':emo_label, 'face_position':face_position}
                print(emotion_data)
                
    else:
        emotion_data = {'emotion_label': "", 'face_position': []}

    return emotion_data        
            
        
def create_hand_data(img):

    image_height, image_width, _ = np.shape(img)
    # 转换为RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 得到检测结果
    results = hands.process(imgRGB)
    hand_data = {'gesture' : ""}
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        # 采集所有关键点的坐标
        list_lms = []
        for i in range(21):
            pos_x = hand.landmark[i].x * image_width
            pos_y = hand.landmark[i].y * image_height
            list_lms.append([int(pos_x), int(pos_y)])

        # 构造凸包点
        list_lms = np.array(list_lms, dtype=np.int32)
        hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
        hull = cv2.convexHull(list_lms[hull_index, :])

        # 查找外部的点数
        n_fig = -1
        ll = [4, 8, 12, 16, 20]
        up_fingers = []

        for i in ll:
            pt = (int(list_lms[i][0]), int(list_lms[i][1]))
            dist = cv2.pointPolygonTest(hull, pt, True)
            if dist < 0:
                up_fingers.append(i)

        str_guester = get_str_guester(up_fingers, list_lms)

        hand_data = {'gesture': str_guester}
    print(hand_data)
    return hand_data
            

def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    return angle

# noinspection PyShadowingNames
def get_str_guester(up_fingers: object, list_lms: object) -> object:
    if len(up_fingers) == 1 and up_fingers[0] == 8:

        v1 = list_lms[6] - list_lms[7]
        v2 = list_lms[8] - list_lms[7]

        angle = get_angle(v1, v2)

        if angle < 160:
            str_guested = "9"
        else:
            str_guested = "1"

    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        str_guested = "good"

    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        str_guested = "bad"

    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        str_guested = "f"

    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_guested = "2"
       # serial.write("2".encode())
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        str_guested = "6"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        str_guested = "8"

    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        str_guested = "3"
       # serial.write("3".encode())
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:

        dis_8_12 = list_lms[8, :] - list_lms[12, :]
        dis_8_12 = np.sqrt(np.dot(dis_8_12, dis_8_12))

        dis_4_12 = list_lms[4, :] - list_lms[12, :]
        dis_4_12 = np.sqrt(np.dot(dis_4_12, dis_4_12))

        if dis_4_12 / (dis_8_12 + 1) < 3:
            # noinspection PyShadowingNames
            str_guested = "7"

        elif dis_4_12 / (dis_8_12 + 1) > 5:
            str_guested = "8"
        else:
            str_guested = "7"

    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        str_guested = "ROCK"

    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[
        3] == 20:
        str_guested = "4"
       # serial.write("4".encode())
    elif len(up_fingers) == 5:
        str_guested = "5"
       # serial.write("5".encode())
    elif len(up_fingers) == 0:
        str_guested = "0"
       # serial.write("0".encode())
    else:
        str_guested = ""

    return str_guested



ct = threading.Thread(target=create_data)
ct.setDaemon(True)
ct.start()

# 线程处理函数，用于发送数据
def send_to_client():
    last_data = None
    last_timestamp = -1
    while True:
        try:
            request = request_queue.get()
            # print(f"request is {request}")
            if request is None:
                # 遇到 None 时退出线程
                break
            if request["type"] == "fetch":
                while not data_queue.empty():
                    last_data = data_queue.get()
                    last_timestamp = request["timestamp"]

                payload = {
                    "type": "reply",
                    "data": last_data,
                    "timestamp": last_timestamp,
                }
                client_socket.send(json.dumps(payload).encode())
            elif request["type"] == "cmd":
                if request["cmd"] == "exit":
                    # 处理退出信号
                    break
                else:
                    print("未知命令：", request["cmd"])
        except (ConnectionResetError, BrokenPipeError):
            print("客户端断开连接。")
            break
        except Exception as e:
            print("发生异常：", e)
            break

# 启动发送数据线程
t = threading.Thread(target=send_to_client)
t.setDaemon(True)
t.start()

while True:
    print("等待客户端连接...")
    client_socket, addr = server_socket.accept()
    print("连接地址：", addr)

    # 线程处理函数，用于处理连接数据
    def handle_client():
        while True:
            try:
                # 接收客户端发送来的数据
                request = client_socket.recv(1024).decode()
                # print(request.strip())
                payload = json.loads(request.strip())

                if payload["type"] == "cmd" and payload["cmd"] == "exit":
                    # 处理退出信号
                    request_queue.put(None)
                    break
                else:
                    request_queue.put(payload)
            except (ConnectionResetError, BrokenPipeError):
                print("客户端断开连接。")
                request_queue.put(None)
                break
            except Exception as e:
                print("发生异常：", e)
                traceback.print_exc()
                request_queue.put(None)
                break
        client_socket.close()

    t = threading.Thread(target=handle_client)
    t.setDaemon(True)
    t.start()
