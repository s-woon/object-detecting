import os
import sys
import cv2
from time import sleep

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from pytube import YouTube

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

save_dir = './video'

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    Frame = pyqtSignal(np.ndarray)

    def run(self):
        global cap
        cap = cv2.VideoCapture(source)

        while True:
            ret, frame = cap.read()
            if ret:
                scale_percent = 50

                # print(frame.shape)
                # calculate the 50 percent of original dimensions
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)

                # dsize
                dsize = (width, height)

                # resize image
                img =  cv2.resize(frame, dsize)
                fps = cap.get(cv2.CAP_PROP_FPS)
                # delay = round(1000/fps)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cvc = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = cvc.scaled(1280, 720, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.Frame.emit(img)

                sleep(0.028)
            else:
                print('Done')
                break

class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('main3.ui', self)

        self.th = Thread(self)

        self.saveBtn.clicked.connect(self.videosave)
        self.openBtn.clicked.connect(self.videoopen)
        self.deleteBtn.clicked.connect(self.videodelete)
        self.startBtn.clicked.connect(self.videostart)
        self.stopBtn.clicked.connect(self.videostop)

        self.getimgBtn.clicked.connect(self.getimg)
        self.learningBtn.clicked.connect(self.learning)
        self.dnwBtn.clicked.connect(self.dnw)

        self.actionOpen.triggered.connect(self.videoopen)

# 비디오 재생관리
    def video(self, image):
        self.videoLb.setPixmap(QPixmap.fromImage(image).scaled(self.videoLb.size(), Qt.KeepAspectRatio))

    def getimg(self):
        global source
        if self.lbox.currentItem() == None:
            QMessageBox.warning(self, '오류', '동영상 파일이 없습니다.  ')
        else:
            source = self.lbox.currentItem().text()
            if source == '' or None:
                QMessageBox.warning(self, '오류', '재생할 동영상 파일이 선택되지 않았습니다.  ')
            else:
                print("Get person img start")
                PATH = './person_imgs'

                try:
                    if not os.path.exists(PATH):
                        os.mkdir(PATH)
                except OSError:
                    print("Error: Failed to create the directory.")

                weight = './yolov3.weights'
                cfg = './yolov3.cfg'
                net = cv2.dnn.readNet(weight, cfg)
                classes = None
                with open('./yolov3.txt', 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                for i in range(150):
                    sec = int(60 + i * 2)
                    video = cv2.VideoCapture(source)
                    video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    success, img = video.read()

                    if success:
                        Width = img.shape[1]
                        Height = img.shape[0]
                        boxes = []
                        confidences = []
                        img_rescale = cv2.resize(img, (224, 224))
                        scale = 0.00392
                        blob = cv2.dnn.blobFromImage(img_rescale, scale, (416, 416), (0, 0, 0), True, crop=False)
                        net.setInput(blob)
                        path = './person_imgs/' + 'video1_frame' + '_' + str(i) + '_'
                        conf_t = 0.3
                        nms_t = 0.4
                        layer_names = net.getLayerNames()
                        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                        outs = net.forward(output_layers)
                        for out in outs:
                            for detection in out:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                if class_id == 0:
                                    confidence = scores[class_id]
                                    if confidence > conf_t:
                                        center_x = int(detection[0] * Width)
                                        center_y = int(detection[1] * Height)
                                        w = int(detection[2] * Width)
                                        h = int(detection[3] * Height)
                                        x = center_x - w / 2
                                        y = center_y - h / 2
                                        confidences.append(float(confidence))
                                        boxes.append([x, y, w, h])
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_t, nms_t)
                        boxes = [boxes[i] for i in indices]
                        box_fin = [[x, y, x+w, y+h] for [x, y, w, h] in boxes]
                        imgs = [img[round(y1):round(y2), round(x1):round(x2)] for [x1, y1, x2, y2] in box_fin]
                        if len(imgs) != 0:
                            for i in range(len(imgs)):
                                try:
                                    cv2.imwrite(path + str(i) + '.jpg', imgs[i])
                                except:
                                    pass
                print("Get person img finish")

    def learning(self):
        # print(len(os.listdir('./person_imgs')))

        mobilenet = MobileNetV2(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet')

        mobilenet.trainable = True
        for i in range(len(mobilenet.layers)):
            mobilenet.layers[i].trainable = False

        model = Sequential()
        model.add(mobilenet)
        model.add(MaxPooling2D(3))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(6, activation='softmax')) # number of classes = 6
        model.summary()

        PATH = './person_imgs/'
        img_list = os.listdir(PATH)
        # print(len(img_list))
        labels = []
        DATA = []


        for i in range(len(img_list)):
            img = load_img(PATH + img_list[i], target_size=(128, 128))
            img = img_to_array(img)
            img = preprocess_input(img)
            DATA.append(img)

        data = np.array([])
        for i in range(len(img_list)):
            data_folder = data
            data = np.append(data, data_folder)
            label = len(data_folder) * [img_list[i]]
            labels.extend(label)

        data = data.reshape(-1, 128, 128, 3)
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)

        (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, stratify=labels)

        print(x_train.shape, x_test.shape)


    def dnw(self):
        pass

    def progress_function(self, stream, chunk, bytes_remaining):
        size = self.video.filesize
        progress = ((float(abs(bytes_remaining - size) / size)) * float(100))
        self.progressBar.setValue(progress)
        if self.progressBar.value() == 100:
            QMessageBox.warning(self, '완료', '동영상파일 다운로드 완료!  ')
            self.progressBar.setValue(0)

    def videosave(self):
        source = self.downurl.text()
        yt = YouTube(source, on_progress_callback=self.progress_function)
        self.video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        self.video.download(save_dir)

    def videoopen(self):
        fname = QFileDialog.getOpenFileName(self)
        self.lbox.addItem(fname[0])

    def videodelete(self):
        index = self.lbox.currentRow()
        self.lbox.takeItem(index)

    def videostart(self):
        global source
        if self.lbox.currentItem() == None:
            QMessageBox.warning(self, '오류', '재생할 동영상 파일이 없습니다.  ')
        else:
            source = self.lbox.currentItem().text()
            if source == '' or None:
                QMessageBox.warning(self, '오류', '재생할 동영상 파일이 선택되지 않았습니다.  ')
            else:
                self.th.start()
                self.th.changePixmap.connect(self.video)

    def videostop(self):
        if self.th.isRunning():
            self.th.terminate()
            self.th.changePixmap.disconnect(self.video)
        else:
            QMessageBox.warning(self, '오류', '재생중인 동영상이 없습니다.')


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
