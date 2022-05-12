
import sys
import cv2
from time import sleep
from copy import deepcopy
import colorsys

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QColorDialog, QDialog
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from pytube import YouTube
from PIL import ImageColor
import pyautogui
from PIL import ImageGrab

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

hsv = 0

save_dir = './video'

sys.path.append("../..")
PATH_TO_CKPT = '../model/frozen_inference_graph.pb'
PATH_TO_LABELS = '../data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)




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

        self.selectcolorBtn.clicked.connect(self.selectcolor)

        self.detectBtn.clicked.connect(self.detectstart)

        self.actionOpen.triggered.connect(self.videoopen)

    def rangeFor(self, seg):
        seg.sort()
        rangeC = []
        st = 0
        for i in range(len(seg) - 1):
            if (seg[i + 1] - seg[i] > 0.015):
                rangeC.append([seg[st] - 0.005, seg[i] + 0.005])
                st = i + 1
        if st != len(seg) - 1 and len(seg) != 0:
            rangeC.append([seg[st] - 0.005, seg[i] + 0.005])
        return rangeC

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

    def nothing(self, x):
        pass

    def selectcolor(self):
        filename = QFileDialog.getOpenFileName(self)
        filesource = filename[0]

        # cv2.setMouseCallback('img_color', self.mouse_callback)
        self.img_color = cv2.imread(filesource)
        image = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.t1colorLb.setPixmap(QPixmap.fromImage(image).scaled(self.t1colorLb.size(), Qt.KeepAspectRatio))


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

    def giveRange(self, count, threshold, width):
        indices = np.argwhere(count > threshold)
        rangeC = np.array([[0, 0]])
        for k in range(0, len(indices) - 1):
            left = right = indices[k][0]
            flagL = flagR = False
            for i in range(0, len(count) - 1):

                if (indices[k][0] - i > 0 and count[indices[k][0] - i] > width and not flagL):
                    left = indices[k][0] - i
                else:
                    flagL = True

                if (indices[k][0] + i < len(count) and count[indices[k][0] + i] > width and not flagR):
                    right = indices[k][0] + i
                else:
                    flagR = True

                if (flagL and flagR):
                    rangeC = np.append(rangeC, [[left / 256, right / 256]], axis=0)
                    break
        rangeC = np.delete(rangeC, 0, 0)
        return rangeC

    def segmentsIn(self, rangeC):
        seg = []
        for i in range(0, len(rangeC)):
            start = round(rangeC[i][0], 2)
            end = round(rangeC[i][1], 2)
            seg = np.append(seg, [start, end])
            for j in range(1, int(100 * (end - start))):
                seg = np.append(seg, round(start + 0.01 * j, 2))
        return list(set(seg))

    def detectstart(self):
        global source

        seg = []
        features = []
        playerIndices = []

        hsv1 = []
        hsv2 = []

        team1 = []
        team2 = []

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

                # print(self.t1name, self.t2name)
                # t2color = self.colorfinder(t2color)
                global cap
                cap = cv2.VideoCapture(source)

                with detection_graph.as_default():
                    with tf.compat.v1.Session(graph=detection_graph) as sess:
                        counter = 0
                        while (True):
                            ret, image_np = cap.read()
                            counter += 1
                            if ret:
                                h = image_np.shape[0]
                                w = image_np.shape[1]

                            if not ret:
                                break
                            if counter % 1 == 0:
                                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                                image_np_expanded = np.expand_dims(image_np, axis=0)
                                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                                # Each box represents a part of the image where a particular object was detected.
                                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                                # Each score represent how level of confidence for each of the objects.
                                # Score is shown on the result image, together with the class label.
                                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                                # Actual detection.
                                (boxes, scores, classes, num_detections) = sess.run(
                                    [boxes, scores, classes, num_detections],
                                    feed_dict={image_tensor: image_np_expanded})
                                # Visualization of the results of a detection.
                                vis_util.visualize_boxes_and_labels_on_image_array(
                                    image_np,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=3,
                                    min_score_thresh=0.6)

                                frame_number = counter
                                loc = {}
                                for n in range(len(scores[0])):
                                    if scores[0][n] > 0.60:
                                        # Calculate position
                                        ymin = int(boxes[0][n][0] * h)
                                        xmin = int(boxes[0][n][1] * w)
                                        ymax = int(boxes[0][n][2] * h)
                                        xmax = int(boxes[0][n][3] * w)

                                        # Find label corresponding to that class
                                        for cat in categories:
                                            if cat['id'] == classes[0][n]:
                                                label = cat['name'] # 여기까지 사람추출해내고 라벨에 person 대입

                                        ## extract every person
                                        # if label == 'person':

                                        crop_img = image_np[ymin:ymax, xmin:xmax]

                                        img = cv2.resize(crop_img, dsize=(50, 100), interpolation=cv2.INTER_LINEAR)
                                        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                                        H = HSV[:, :, 0]
                                        S = HSV[:, :, 1]
                                        V = HSV[:, :, 2]

                                        countH, _ = np.histogram(H, 256)
                                        countS, _ = np.histogram(S, 256)
                                        countV, _ = np.histogram(V, 256)

                                        meanH = np.mean(H)
                                        meanS = np.mean(S)
                                        meanV = np.mean(V)
                                        medianH = np.median(H)
                                        medianS = np.median(S)
                                        medianV = np.median(V)

                                        features.append([meanH, meanS, meanV, medianH, medianS, medianV])

                                        rangeH = self.giveRange(countH, 15, 10)
                                        rangeS = self.giveRange(countS, 5, 4)
                                        rangeV = self.giveRange(countV, 5, 3)

                                        segH = self.segmentsIn(rangeH)
                                        segS = self.segmentsIn(rangeS)
                                        segV = self.segmentsIn(rangeV)

                                        seg.append([segH, segS, segV])


                                features = np.array(features).astype(np.float32)
                                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                                flags = cv2.KMEANS_RANDOM_CENTERS
                                _, idx, _ = cv2.kmeans(features, 2, None, criteria, 15, flags)
                                print(idx)
                                hsv1 = []
                                hsv2 = []
                                print(seg)
                                for i in range(0, len(seg)):
                                    team1 = []
                                    team2 = []
                                    for j in range(0, len(idx)):
                                        if idx[j]:
                                            team1 = team1 + seg[j][i]
                                        else:
                                            team2 = team2 + seg[j][i]

                                    hsv1.append(team1)
                                    hsv2.append(team2)

                                # removing intersections
                                print("ssss ", hsv1)
                                print("zzzz ", hsv2)
                                sameH = set(hsv1[0]).intersection(hsv2[0])
                                setH1 = list(set(hsv1[0]) - sameH)
                                setH2 = list(set(hsv2[0]) - sameH)
                                rangeSame = self.rangeFor(list(sameH))

                                for i in range(0, len(rangeSame)):
                                    cnt1 = cnt2 = team1 = team2 = 0
                                    mask_i = np.zeros(crop_img.shape)
                                    mask_i[crop_img] = 1
                                    mask_i = mask_i.astype(np.uint8)
                                    mask_3i = np.squeeze(np.stack((mask_i,) * 3, -1))
                                    player = np.multiply(img, mask_3i)

                                H = HSV[:, :, 0]

                                mask = np.zeros(crop_img.shape)
                                mask[(H > rangeSame[i][0]) & (H < rangeSame[i][1])] = 1
                                if idx[j]:
                                    team1 += 1
                                    cnt1 += mask.sum()
                                else:
                                    team2 += 1
                                    cnt2 += mask.sum()

                                if (cnt1 / team1) > (cnt2 / team2):
                                    setH1 = np.append(setH1, self.segmentsIn([rangeSame[i]]))
                                else:
                                    setH2 = np.append(setH2, self.segmentsIn([rangeSame[i]]))

                                rangeH1 = self.rangeFor(setH1)
                                rangeH2 = self.rangeFor(setH2)
                                rangeS1 = self.rangeFor(hsv1[1])
                                rangeS2 = self.rangeFor(hsv2[1])
                                rangeV1 = self.rangeFor(hsv1[2])
                                rangeV2 = self.rangeFor(hsv2[2])

                                print(rangeH1, rangeS1, rangeV1, rangeH2, rangeS2, rangeV2)

                                for i in range(len(idx)):
                                    if idx[0][i] == int(1):
                                        cv2.putText(image_np, self.t1name, (xmin - 2, ymin - 2),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                                    else:
                                        cv2.putText(image_np, self.t2name, (xmin - 2, ymin - 2),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

                            cv2.imshow('image', image_np)

                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                cap.release()
                                break


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
            print(rgb, 'rgb')

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
            print(rgb, 'rgb')

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


