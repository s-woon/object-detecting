import os
import sys
import cv2
from time import sleep

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QDialog
from PyQt5.uic import loadUi
from pytube import YouTube
import pyautogui
from PIL import ImageGrab

import readyolo

hsv = 0

save_dir = './video'

# YOLO 네트워크 불러오기
weight = './yolov3.weights'
cfg = './yolov3.cfg'
net = cv2.dnn.readNet(weight, cfg)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = None
with open('../yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

boxes = []
confidences = []
conf_threshold = 0.3
nms_threshold = 0.4

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    ndimg = pyqtSignal(np.ndarray)

    def run(self):
        global cap
        cap = cv2.VideoCapture(source)

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
                cvc = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = cvc.scaled(1280, 720, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.ndimg.emit(img)

                sleep(0.028)
            else:
                print('Done')
                break

''' 메인윈도우 '''
class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('main5.ui', self)
        self.th = Thread(self)
        self.i = 0

        self.saveBtn.clicked.connect(self.videosave)
        self.openBtn.clicked.connect(self.videoopen)
        self.deleteBtn.clicked.connect(self.videodelete)
        self.startBtn.clicked.connect(self.videostart)
        self.stopBtn.clicked.connect(self.videostop)
        self.captureBtn.clicked.connect(self.captureslot)


        self.t1settingBtn.clicked.connect(self.t1setting)
        self.t2settingBtn.clicked.connect(self.t2setting)

        self.croppersonBtn.clicked.connect(self.crop_personimg)

        self.detectBtn.clicked.connect(self.detectstart)

        self.actionOpen.triggered.connect(self.videoopen)

    def t1setting(self):
        dlg = t1SettingDialog()
        dlg.exec_()
        self.t1name = dlg.tname
        self.t1color = dlg.tcolor

        self.t1lowerhsv = dlg.lowerhsv
        self.t1upperhsv = dlg.upperhsv

        self.t1nameLE.setText(self.t1name)
        self.t1colorLE.setText(self.t1color)

    def t2setting(self):
        dlg = t2SettingDialog()
        dlg.exec_()
        self.t2name = dlg.tname
        self.t2color = dlg.tcolor

        self.t2lowerhsv = dlg.lowerhsv
        self.t2upperhsv = dlg.upperhsv

        self.t2nameLE.setText(self.t2name)
        self.t2colorLE.setText(self.t2color)

# 비디오 재생관리
    def video(self, image):
        self.videoLb.setPixmap(QPixmap.fromImage(image).scaled(self.videoLb.size(), Qt.KeepAspectRatio))

    def captureslot(self):
        self.th.ndimg.connect(self.capture)

    def capture(self, image):
        cv2.imwrite("./capture/cap" + str(self.i) + ".png", image)
        print("./capture/cap" + str(self.i) + ".png")
        self.th.ndimg.disconnect(self.capture)
        self.i += 1

    def count_nonblack_np(self, img):
        """Return the number of pixels in img that are not black.
        img must be a Numpy array with colour values along the last axis."""
        return img.any(axis=-1).sum()

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
        global source
        if self.lbox.currentItem() == None:
            QMessageBox.warning(self, '오류', '재생할 동영상 파일이 없습니다.  ')
        else:
            source = self.lbox.currentItem().text()
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
                # writer = cv2.VideoWriter(source + '_detecting.avi', fourcc, fps, (640, 360))
                print(self.t1nameLE.text(), self.t2nameLE.text())
                global cap


                # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

                # 영상 불러오기
                cap = cv2.VideoCapture(source)

                while cap.isOpened():
                    ret, frame = cap.read()
                    boxes, confidences, class_ids = readyolo.yolo(frame=frame, net=net, output_layers=output_layers)

                    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
                    print(f"boxes: {boxes}")
                    print(f"confidences: {confidences}")

                    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

                    # 후보 박스 중 선택된 박스의 인덱스 출력
                    print(f"indexes: ", end='')
                    for index in indexes:
                        print(index, end=' ')
                    print("\n\n============================== classes ==============================")

                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            class_name = classes[class_ids[i]]
                            label = f"{class_name} {confidences[i]:.2f}"
                            color = COLORS[class_ids[i]]

                            if class_name == 'person':
                                # 사각형 테두리 그리기 및 텍스트 쓰기
                                crop_img = frame[y:y+h, x:x+w]
                                hists = []
                                img = cv2.resize(crop_img, dsize=(50, 100), interpolation=cv2.INTER_LINEAR)
                                # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                                bgr = ['b', 'g', 'r']
                                for i in range(len(bgr)):
                                    hist = cv2.calcHist([img], [i], None, [4], [0, 256])
                                    hists.append(hist)

                                histogram = np.concatenate(hists)
                                histogram = cv2.normalize(histogram, histogram)

                                print(histogram)

                    cv2.imshow('aaa', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        cap.release()


    def nparr2qimg(self, cvimg):
        ''' convert cv2 bgr image -> rgb qimg '''
        h, w, c = cvimg.shape
        byte_per_line = w * c  # cvimg.step() #* step # NOTE:when image format problem..
        return QImage(cvimg.data, w, h, byte_per_line,
                      QImage.Format_RGB888).rgbSwapped()

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


''' t1 Dialog '''
class t1SettingDialog(QDialog):
    def __init__(self):
        super(t1SettingDialog, self).__init__()
        loadUi('dialog.ui', self)

        self.saveBtn.clicked.connect(self.save)
        self.cancelBtn.clicked.connect(self.cancel)

        self.openpictureBtn.clicked.connect(self.selectpicture)

    def save(self):
        self.tname = self.tnameLE.text()
        self.tcolor = self.tcolorLE.text()
        self.close()

    def cancel(self):
        self.tname = None
        self.tcolor = None
        self.close()

    def mousePressEvent(self, e):
        if e.buttons():

            screen = ImageGrab.grab()
            rgb = screen.getpixel(pyautogui.position())

            pixel = np.uint8([[rgb]])
            t1hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)
            hsv = t1hsv[0][0]

            # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
            if hsv[0] < 10:
                print("case1")
                lower1 = np.array([hsv[0] - 10 + 180, 30, 30])
                upper1 = np.array([180, 255, 255])
                lower2 = np.array([0, 30, 30])
                upper2 = np.array([hsv[0], 255, 255])
                lower3 = np.array([hsv[0], 30, 30])
                upper3 = np.array([hsv[0] + 10, 255, 255])
                #     print(i-10+180, 180, 0, i)
                #     print(i, i+10)
                self.lowerhsv = lower3
                self.upperhsv = upper1

            elif hsv[0] > 170:
                print("case2")
                lower1 = np.array([hsv[0], 30, 30])
                upper1 = np.array([180, 255, 255])
                lower2 = np.array([0, 30, 30])
                upper2 = np.array([hsv[0] + 10 - 180, 255, 255])
                lower3 = np.array([hsv[0] - 10, 30, 30])
                upper3 = np.array([hsv[0], 255, 255])
                #     print(i, 180, 0, i+10-180)
                #     print(i-10, i)
                self.lowerhsv = lower3
                self.upperhsv = upper1

            else:
                print("case3")
                lower1 = np.array([hsv[0], 30, 30])
                upper1 = np.array([hsv[0] + 10, 255, 255])
                lower2 = np.array([hsv[0] - 10, 30, 30])
                upper2 = np.array([hsv[0], 255, 255])
                lower3 = np.array([hsv[0] - 10, 30, 30])
                upper3 = np.array([hsv[0], 255, 255])
                #     print(i, i+10)
                #     print(i-10, i)
                self.lowerhsv = lower3
                self.upperhsv = upper1

            print(self.lowerhsv, self.upperhsv, 'lower, upper')
            self.tcolorLE.setText(str(self.lowerhsv))


    def selectpicture(self):
        filename = QFileDialog.getOpenFileName(self)
        filesource = filename[0]
        self.img_color = cv2.imread(filesource)
        image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.tpictureLb.setPixmap(QPixmap.fromImage(image).scaled(self.tpictureLb.size(), Qt.KeepAspectRatio))


''' t2 Dialog '''
class t2SettingDialog(QDialog):
    def __init__(self):
        super(t2SettingDialog, self).__init__()
        loadUi('dialog.ui', self)

        self.saveBtn.clicked.connect(self.save)
        self.cancelBtn.clicked.connect(self.cancel)

        self.openpictureBtn.clicked.connect(self.selectpicture)

    def save(self):
        self.tname = self.tnameLE.text()
        self.tcolor = self.tcolorLE.text()
        self.close()

    def cancel(self):
        self.tname = None
        self.tcolor = None
        self.close()

    def mousePressEvent(self, e):
        if e.buttons():
            screen = ImageGrab.grab()
            rgb = screen.getpixel(pyautogui.position())

            pixel = np.uint8([[rgb]])
            t1hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)
            hsv = t1hsv[0][0]

            # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
            if hsv[0] < 10:
                print("case1")
                lower1 = np.array([hsv[0] - 10 + 180, 30, 30])
                upper1 = np.array([180, 255, 255])
                lower2 = np.array([0, 30, 30])
                upper2 = np.array([hsv[0], 255, 255])
                lower3 = np.array([hsv[0], 30, 30])
                upper3 = np.array([hsv[0] + 10, 255, 255])
                #     print(i-10+180, 180, 0, i)
                #     print(i, i+10)
                self.lowerhsv = lower3
                self.upperhsv = upper1

            elif hsv[0] > 170:
                print("case2")
                lower1 = np.array([hsv[0], 30, 30])
                upper1 = np.array([180, 255, 255])
                lower2 = np.array([0, 30, 30])
                upper2 = np.array([hsv[0] + 10 - 180, 255, 255])
                lower3 = np.array([hsv[0] - 10, 30, 30])
                upper3 = np.array([hsv[0], 255, 255])
                #     print(i, 180, 0, i+10-180)
                #     print(i-10, i)
                self.lowerhsv = lower3
                self.upperhsv = upper1

            else:
                print("case3")
                lower1 = np.array([hsv[0], 30, 30])
                upper1 = np.array([hsv[0] + 10, 255, 255])
                lower2 = np.array([hsv[0] - 10, 30, 30])
                upper2 = np.array([hsv[0], 255, 255])
                lower3 = np.array([hsv[0] - 10, 30, 30])
                upper3 = np.array([hsv[0], 255, 255])
                #     print(i, i+10)
                #     print(i-10, i)
                self.lowerhsv = lower3
                self.upperhsv = upper1

            print(self.lowerhsv, self.upperhsv, 'lower, upper')
            self.tcolorLE.setText(str(self.lowerhsv))

    def selectpicture(self):
        filename = QFileDialog.getOpenFileName(self)
        filesource = filename[0]
        self.img_color = cv2.imread(filesource)
        image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.tpictureLb.setPixmap(QPixmap.fromImage(image).scaled(self.tpictureLb.size(), Qt.KeepAspectRatio))

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()


