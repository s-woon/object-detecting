
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

    def count_nonblack_np(self, img):
        """Return the number of pixels in img that are not black.
        img must be a Numpy array with colour values along the last axis."""
        return img.any(axis=-1).sum()

    def detect_team(self, image, show=True):
        # define the list of boundaries
        # print(self.lowerhsv, self.upperhsv, 'detect lower, upper')
        boundaries = [
            (self.t1lowerhsv, self.t1upperhsv),
            (self.t2lowerhsv, self.t2upperhsv)
        ]
        i = 0
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask=mask)
            tot_pix = self.count_nonblack_np(image)
            color_pix = self.count_nonblack_np(output)
            ratio = color_pix / tot_pix
            print("ratio is:", ratio)
            # print(self.t1color, self.t2color, 'self.t1color, self.t2color')
            if ratio > 0.01 and i == 0:
                return self.t1color
            elif ratio > 0.01 and i == 1:
                return self.t2color

            i += 1

            if show == True:
                cv2.imshow("images", np.hstack([image, output]))
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
        return 'not_sure'

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

                print(self.t1name, self.t2name)
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
                                        if label == 'person':
                                            # crop them
                                            crop_img = image_np[ymin:ymax, xmin:xmax]
                                            player_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                                            mask1 = cv2.inRange(player_hsv, self.t1lowerhsv, self.t1upperhsv)
                                            res1 = cv2.bitwise_and(crop_img, crop_img, mask=mask1)
                                            res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                                            res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                                            nzCount = cv2.countNonZero(res1)

                                            mask2 = cv2.inRange(player_hsv, self.t2lowerhsv, self.t2upperhsv)
                                            res2 = cv2.bitwise_and(crop_img, crop_img, mask=mask2)
                                            res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                                            res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                                            nzCountred = cv2.countNonZero(res2)

                                            if (nzCount >= 20):
                                                # Mark blue jersy players as france
                                                cv2.putText(image_np, self.t1name, (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
                                                            cv2.LINE_AA)
                                                cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
                                            else:
                                                pass
                                            if (nzCountred >= 20):
                                                # Mark red jersy players as belgium
                                                cv2.putText(image_np, self.t2name, (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                                            cv2.LINE_AA)
                                                cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                                            else:
                                                pass


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


