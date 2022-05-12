
import sys
import cv2
from time import sleep, time

import imutils
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from pytube import YouTube
from pytube.cli import on_progress


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
        loadUi('main2.ui', self)

        self.th = Thread(self)

        self.saveBtn.clicked.connect(self.videosave)
        self.openBtn.clicked.connect(self.videoopen)
        self.deleteBtn.clicked.connect(self.videodelete)
        self.startBtn.clicked.connect(self.videostart)
        self.stopBtn.clicked.connect(self.videostop)

        self.detectBtn.clicked.connect(self.detectstart)

        self.actionOpen.triggered.connect(self.videoopen)

# 비디오 재생관리
    def video(self, image):
        self.videoLb.setPixmap(QPixmap.fromImage(image).scaled(self.videoLb.size(), Qt.KeepAspectRatio))

    def detectstart(self):
        global source
        if self.lbox.currentItem() == None:
            QMessageBox.warning(self, '오류', '재생할 동영상 파일이 없습니다.  ')
        else:
            source = self.lbox.currentItem().text()
            if source == '' or None:
                QMessageBox.warning(self, '오류', '재생할 동영상 파일이 선택되지 않았습니다.  ')
            else:
                print("Detecting start")
                global writer
                fps = 29.97
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                writer = cv2.VideoWriter(source + '_detecting.avi', fourcc, fps, (640, 360))

                weight = './yolov3.weights'
                cfg = './yolov3.cfg'
                net = cv2.dnn.readNet(weight, cfg)
                classes = None
                with open('../yolov3.txt', 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))

                vid = cv2.VideoCapture(source)
                while vid:
                    ret, img = vid.read()
                    j = 0
                    if ret:
                        # try:
                        #     prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                        #         else cv2.CAP_PROP_FRAME_COUNT
                        #     total = int(vid.get(prop))
                        #     print("[INFO] {} total frames in video".format(total))
                        # except:
                        #     print("[INFO] could not determine # of frames in video")
                        #     print("[INFO] no approx. completion time can be provided")
                        prop = cv2.CAP_PROP_FRAME_COUNT
                        total = int(vid.get(prop))
                        # try:
                        #     # img = cv2.resize(img, None, fx=0.4, fy=0.4)
                        #     img = cv2.resize(img, (416, 416))
                        # except Exception as e:
                        #     print(str(e))
                        height, width, channels = img.shape

                        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                        net.setInput(blob)
                        outs = net.forward(output_layers)
                        class_ids = []
                        confidences = []
                        boxes = []

                        for out in outs:
                            for detection in out:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]
                                if confidence > 0.5:
                                    # Object detected
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)
                                    # 좌표
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)
                                    boxes.append([x, y, w, h])
                                    confidences.append(float(confidence))
                                    class_ids.append(class_id)

                        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                        font = cv2.FONT_HERSHEY_PLAIN
                        for i in range(len(boxes)):
                            if i in indexes:
                                x, y, w, h = boxes[i]
                                label = str(classes[class_ids[i]])
                                color = colors[i]
                                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(img, label, (x, y - 10), font, 1, color, 3)

                        writer.write(img)
                        j += 1
                        print("[INFO] {} total frames in video {}/{}".format(total, j, total))
                writer.release()
                exit()

    def nparr2qimg(self, cvimg):
        ''' convert cv2 bgr image -> rgb qimg '''
        h, w, c = cvimg.shape
        byte_per_line = w * c  # cvimg.step() #* step # NOTE:when image format problem..
        return QImage(cvimg.data, w, h, byte_per_line,
                      QImage.Format_RGB888).rgbSwapped()

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
