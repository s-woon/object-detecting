import os
import sys
import datetime

import cv2
from time import sleep

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QDialog
from PyQt5.uic import loadUi
from pytube import YouTube

from media import CMultiMedia
import readyolo
from camera import Camera

hsv = 0

save_dir = './video'

# YOLO 네트워크 불러오기
weight = './yolov3.weights'
cfg = './yolov3.cfg'
net = cv2.dnn.readNet(weight, cfg)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = None
with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

boxes = []
confidences = []
conf_threshold = 0.3
nms_threshold = 0.4

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    Frame = pyqtSignal(np.ndarray)

    def run(self):
        global cap
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if ret:
                scale_percent = 50
                # print(frame.shape)
                # calculate the 50 percent of original dimensions
                width = int(frame.shape[1])
                height = int(frame.shape[0])

                # dsize
                dsize = (width, height)

                # resize image
                img = cv2.resize(frame, dsize)
                fps = cap.get(cv2.CAP_PROP_FPS)
                # delay = round(1000/fps)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbImage = cv2.flip(rgbImage, 1)
                cvc = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = cvc.scaled(1280, 720, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.Frame.emit(rgbImage)

                sleep(0.028)
            else:
                print('Done')
                break

''' 메인윈도우 '''
class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('main2.ui', self)
        self.th = Thread(self)
        self.i = 0
        self.t1name = None
        self.t1hist = None
        self.t1image = None
        self.t2name = None
        self.t2hist = None
        self.t2image = None

        # Multimedia Object
        self.mp = CMultiMedia(self, self.view)

        # video background color
        pal = QPalette()
        pal.setColor(QPalette.Background, Qt.black)
        self.view.setAutoFillBackground(True);
        self.view.setPalette(pal)

        # volume, slider
        self.vol.setRange(0, 100)
        self.vol.setValue(50)

        # play time
        self.duration = ''

        # signal
        self.saveBtn.clicked.connect(self.videosave)

        self.list.itemDoubleClicked.connect(self.dbClickList)
        self.vol.valueChanged.connect(self.volumeChanged)
        self.bar.sliderMoved.connect(self.barChanged)

        self.btn_add.clicked.connect(self.clickAdd)
        self.btn_del.clicked.connect(self.clickDel)
        self.btn_play.clicked.connect(self.clickPlay)
        self.btn_stop.clicked.connect(self.clickStop)
        self.btn_pause.clicked.connect(self.clickPause)
        self.btn_forward.clicked.connect(self.clickForward)
        self.btn_prev.clicked.connect(self.clickPrev)

        self.t1settingBtn.clicked.connect(self.t1setting)
        self.t2settingBtn.clicked.connect(self.t2setting)

        self.croppersonBtn.clicked.connect(self.crop_personimg)

        self.detectBtn.clicked.connect(self.detectstart)

        # stackedWidget
        self.actionmain.triggered.connect(self.gomain)
        self.actioncamera.triggered.connect(self.gocamera)

        self.stackedWidget.insertWidget(1, Camera(self, self.th))

    def gomain(self):
        self.stackedWidget.setCurrentIndex(0)

    def gocamera(self):
        self.stackedWidget.setCurrentIndex(1)

    def t1setting(self):
        dlg = t1SettingDialog()
        dlg.exec_()
        if dlg.tname and dlg.t1image:
            self.t1name = dlg.tname
            self.t1nameLE.setText(self.t1name)
            self.t1hist = dlg.t1hist
            self.t1image = dlg.t1image
            self.t1imageLB.setPixmap(QPixmap.fromImage(self.t1image).scaled(self.t1imageLB.size(), Qt.KeepAspectRatio))
        else:
            pass

    def t2setting(self):
        dlg = t2SettingDialog()
        dlg.exec_()
        if dlg.tname and dlg.t2image:
            self.t2name = dlg.tname
            self.t2nameLE.setText(self.t2name)
            self.t2hist = dlg.t2hist
            self.t2image = dlg.t2image
            self.t2imageLB.setPixmap(QPixmap.fromImage(self.t2image).scaled(self.t2imageLB.size(), Qt.KeepAspectRatio))
        else:
            pass

# 비디오 재생관리
    def video(self, image):
        self.videoLb.setPixmap(QPixmap.fromImage(image).scaled(self.videoLb.size(), Qt.KeepAspectRatio))

    def get_boxes_coordinate(self, boxes, confidences, conf_t, nms_t):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_t, nms_t)
        boxes = [boxes[i] for i in indices]
        return [[x, y, x + w, y + h] for [x, y, w, h] in boxes]

    def getFrame(self, sec, video_path):
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = video.read()
        return hasFrames, image

    def get_crop_img(self, img, boxes):
        return [img[round(y1):round(y2), round(x1):round(x2)] for [x1, y1, x2, y2] in boxes]

    def get_person_imgs(self, img, net, conf_t, nms_t, path):
        Width = img.shape[1]
        Height = img.shape[0]
        img_rescale = cv2.resize(img, (224, 224))
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(img_rescale, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
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

        box_fin = self.get_boxes_coordinate(boxes, confidences, conf_t, nms_t)
        imgs = self.get_crop_img(img, box_fin)
        if len(imgs) != 0:
            for i in range(len(imgs)):
                try:
                    cv2.imwrite(path + str(i) + '.jpg', imgs[i])
                except:
                    pass

    def createDirectory(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError: print("Error: Failed to create the directory.")

    def crop_personimg(self):
        print('aa')
        global source
        if self.list.currentItem() == None:
            QMessageBox.warning(self, '오류', '재생할 동영상 파일이 없습니다.  ')
        else:
            source = self.list.currentItem().text()
            print(source)
            if source == '' or None:
                QMessageBox.warning(self, '오류', '재생할 동영상 파일이 선택되지 않았습니다.  ')
            else:
                for i in range(30):
                    success, img = self.getFrame(60 + i * 2, source)
                    if success:
                        self.createDirectory('./person_imgs/')
                        path = './person_imgs/' + 'video1_frame' + '_' + str(i) + '_'
                        self.get_person_imgs(img, net, conf_threshold, nms_threshold, path)
                        print('video1_frame' + '_' + str(i) + '_')

    def detectstart(self):
        global source
        if self.list.currentItem() == None:
            QMessageBox.warning(self, '오류', '재생할 동영상 파일이 없습니다.  ')
        elif self.t1name is None or self.t1image is None or self.t2name is None or self.t2image is None:
            QMessageBox.warning(self, '오류', '팀 세팅이 없습니다.  ')
        else:
            source = self.list.currentItem().text()
            if source == '' or None:
                QMessageBox.warning(self, '오류', '재생할 동영상 파일이 선택되지 않았습니다.  ')
            else:
                print("Detecting start")
                QMessageBox.warning(self, 'Detecting & Recording', '객체 탐지 및 녹화를 시작합니다.')
                print(self.t1nameLE.text(), self.t2nameLE.text())
                global cap


                # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

                # 영상 불러오기
                cap = cv2.VideoCapture(source)
                width = int(cap.get(3))
                height = int(cap.get(4))

                global writer
                fps = 29.97
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                writer = cv2.VideoWriter(source + '_detecting.avi', fourcc, fps, (width, height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    boxes, confidences, class_ids = readyolo.yolo(frame=frame, net=net, output_layers=output_layers)

                    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            class_name = classes[class_ids[i]]

                            if class_name == 'person':
                                # 사각형 테두리 그리기 및 텍스트 쓰기
                                crop_img = frame[y:y+h, x:x+w]
                                hists = []
                                try:
                                    img = cv2.resize(crop_img, dsize=(50, 100), interpolation=cv2.INTER_LINEAR)
                                except Exception as e:
                                    print(str(e))

                                for i in range(3):
                                    hist = cv2.calcHist([img], [i], None, [4], [0, 256])
                                    hists.append(hist)

                                histogram = np.concatenate(hists)
                                histogram = cv2.normalize(histogram, histogram)

                                compare = cv2.compareHist(histogram, self.t1hist, cv2.HISTCMP_CHISQR)
                                print(compare, 'compare')
                                # int(x) 값으로 비교값 조절
                                if compare < int(8):
                                    cv2.putText(frame, self.t1name, (x - 2, y - 2),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1,
                                                cv2.LINE_AA)
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                                else:
                                    cv2.putText(frame, self.t2name, (x - 2, y - 2),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1,
                                                cv2.LINE_AA)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

                    cv2.imshow('Detecting', frame)
                    writer.write(frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        writer.release()
                        cap.release()

    def progress_function(self, stream, chunk, bytes_remaining):
        size = self.video.filesize
        progress = ((float(abs(bytes_remaining - size) / size)) * float(100))
        self.downloadbar.setValue(progress)
        if self.downloadbar.value() == 100:
            QMessageBox.warning(self, '완료', '동영상파일 다운로드 완료!  ')
            self.downloadbar.setValue(0)

    def videosave(self):
        source = self.downurl.text()
        yt = YouTube(source, on_progress_callback=self.progress_function)
        self.video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        self.video.download(save_dir)

    def clickAdd(self):
        files, ext = QFileDialog.getOpenFileNames(self, 'Select one or more files to open', '', 'Video (*.mp4 *.mpg *.mpeg *.avi *.wma)')

        if files:
            cnt = len(files)
            row = self.list.count()
            for i in range(row, row + cnt):
                self.list.addItem(files[i - row])
            self.list.setCurrentRow(0)

            self.mp.addMedia(files)

    def clickDel(self):
        row = self.list.currentRow()
        self.list.takeItem(row)
        self.mp.delMedia(row)

    def clickPlay(self):
        index = self.list.currentRow()
        self.mp.playMedia(index)

    def clickStop(self):
        self.mp.stopMedia()

    def clickPause(self):
        self.mp.pauseMedia()

    def clickForward(self):
        cnt = self.list.count()
        curr = self.list.currentRow()
        if curr < cnt - 1:
            self.list.setCurrentRow(curr + 1)
            self.mp.forwardMedia()
        else:
            self.list.setCurrentRow(0)
            self.mp.forwardMedia(end=True)

    def clickPrev(self):
        cnt = self.list.count()
        curr = self.list.currentRow()
        if curr == 0:
            self.list.setCurrentRow(cnt - 1)
            self.mp.prevMedia(begin=True)
        else:
            self.list.setCurrentRow(curr - 1)
            self.mp.prevMedia()

    def dbClickList(self, item):
        row = self.list.row(item)
        self.mp.playMedia(row)

    def volumeChanged(self, vol):
        self.mp.volumeMedia(vol)

    def barChanged(self, pos):
        print(pos)
        self.mp.posMoveMedia(pos)

    def updateState(self, msg):
        self.state.setText(msg)

    def updateBar(self, duration):
        self.bar.setRange(0, duration)
        self.bar.setSingleStep(int(duration / 10))
        self.bar.setPageStep(int(duration / 10))
        self.bar.setTickInterval(int(duration / 10))
        td = datetime.timedelta(milliseconds=duration)
        stime = str(td)
        idx = stime.rfind('.')
        self.duration = stime[:idx]

    def updatePos(self, pos):
        self.bar.setValue(pos)
        td = datetime.timedelta(milliseconds=pos)
        stime = str(td)
        idx = stime.rfind('.')
        stime = f'{stime[:idx]} / {self.duration}'
        self.playtime.setText(stime)

''' t1 Dialog '''
class t1SettingDialog(QDialog):
    def __init__(self):
        super(t1SettingDialog, self).__init__()
        loadUi('dialog.ui', self)
        self.tname = None
        self.t1image = None
        self.t1hist = None
        self.saveBtn.clicked.connect(self.save)
        self.cancelBtn.clicked.connect(self.cancel)

        self.openpictureBtn.clicked.connect(self.selectpicture)

    def save(self):
        self.tname = self.tnameLE.text()
        self.close()

    def cancel(self):
        self.close()

    def selectpicture(self):
        filename = QFileDialog.getOpenFileName(self)
        filesource = filename[0]
        if filesource:
            self.img_color = cv2.imread(filesource)
            img = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
            image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.tpictureLb.setPixmap(QPixmap.fromImage(image).scaled(self.tpictureLb.size(), Qt.KeepAspectRatio))

            hists = []

            img = cv2.resize(img, dsize=(50, 100), interpolation=cv2.INTER_LINEAR)
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [4], [0, 256])
                hists.append(hist)

            histogram = np.concatenate(hists)
            self.t1image = image
            self.t1hist = cv2.normalize(histogram, histogram)


''' t2 Dialog '''
class t2SettingDialog(QDialog):
    def __init__(self):
        super(t2SettingDialog, self).__init__()
        loadUi('dialog.ui', self)
        self.tname = None
        self.t2image = None
        self.t2hist = None
        self.saveBtn.clicked.connect(self.save)
        self.cancelBtn.clicked.connect(self.cancel)

        self.openpictureBtn.clicked.connect(self.selectpicture)

    def save(self):
        self.tname = self.tnameLE.text()
        self.close()

    def cancel(self):
        self.close()

    def selectpicture(self):
        filename = QFileDialog.getOpenFileName(self)
        filesource = filename[0]
        self.img_color = cv2.imread(filesource)
        img = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.tpictureLb.setPixmap(QPixmap.fromImage(image).scaled(self.tpictureLb.size(), Qt.KeepAspectRatio))

        hists = []

        img = cv2.resize(img, dsize=(50, 100), interpolation=cv2.INTER_LINEAR)
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [4], [0, 256])
            hists.append(hist)

        histogram = np.concatenate(hists)
        self.t2image = image
        self.t2hist = cv2.normalize(histogram, histogram)


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()


