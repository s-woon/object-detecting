
import sys
import cv2
from time import sleep
from copy import deepcopy

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QColorDialog
from PyQt5.uic import loadUi
from pytube import YouTube
from PIL import ImageColor

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

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

class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('main5.ui', self)

        self.th = Thread(self)

        self.saveBtn.clicked.connect(self.videosave)
        self.openBtn.clicked.connect(self.videoopen)
        self.deleteBtn.clicked.connect(self.videodelete)
        self.startBtn.clicked.connect(self.videostart)
        self.stopBtn.clicked.connect(self.videostop)
        self.captureBtn.clicked.connect(self.captureslot)

        self.t1colorBtn.clicked.connect(self.selectcolor)


        self.selectcolorBtn.clicked.connect(self.selectcolor)

        self.detectBtn.clicked.connect(self.detectstart)

        self.actionOpen.triggered.connect(self.videoopen)

    def mouse_callback(self, event, x, y, flags, param):
        global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold

        # ????????? ?????? ?????? ????????? ????????? ?????? ???????????? ???????????? HSV??? ???????????????.
        if event == cv2.EVENT_LBUTTONDOWN:
            print(self.img_color[y, x], 'img_color[y, x]')
            color = self.img_color[y, x]

            one_pixel = np.uint8([[color]])
            hsv = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV)
            print(hsv, 'hsv')
            hsv = hsv[0][0]

            threshold = cv2.getTrackbarPos('threshold', 'img_result')

            # HSV ??????????????? ????????? ???????????? ?????? ???????????? ????????? ???????????? ????????? ????????????.
            if hsv[0] < 10:
                print("case1")
                lower_blue1 = np.array([hsv[0] - 10 + 180, threshold, threshold])
                upper_blue1 = np.array([180, 255, 255])
                lower_blue2 = np.array([0, threshold, threshold])
                upper_blue2 = np.array([hsv[0], 255, 255])
                lower_blue3 = np.array([hsv[0], threshold, threshold])
                upper_blue3 = np.array([hsv[0] + 10, 255, 255])
                #     print(i-10+180, 180, 0, i)
                #     print(i, i+10)

            elif hsv[0] > 170:
                print("case2")
                lower_blue1 = np.array([hsv[0], threshold, threshold])
                upper_blue1 = np.array([180, 255, 255])
                lower_blue2 = np.array([0, threshold, threshold])
                upper_blue2 = np.array([hsv[0] + 10 - 180, 255, 255])
                lower_blue3 = np.array([hsv[0] - 10, threshold, threshold])
                upper_blue3 = np.array([hsv[0], 255, 255])
                #     print(i, 180, 0, i+10-180)
                #     print(i-10, i)
            else:
                print("case3")
                lower_blue1 = np.array([hsv[0], threshold, threshold])
                upper_blue1 = np.array([hsv[0] + 10, 255, 255])
                lower_blue2 = np.array([hsv[0] - 10, threshold, threshold])
                upper_blue2 = np.array([hsv[0], 255, 255])
                lower_blue3 = np.array([hsv[0] - 10, threshold, threshold])
                upper_blue3 = np.array([hsv[0], 255, 255])
                #     print(i, i+10)
                #     print(i-10, i)

            print(hsv[0], 'hsv[0]')
            print("@1", lower_blue1, "~", upper_blue1)
            print("@2", lower_blue2, "~", upper_blue2)
            print("@3", lower_blue3, "~", upper_blue3)

    def nothing(self, x):
        pass

    def selectcolor(self):
        filename = QFileDialog.getOpenFileName(self)
        filesource = filename[0]

        cv2.namedWindow('img_color')
        cv2.setMouseCallback('img_color', self.mouse_callback)

        cv2.namedWindow('img_result')
        cv2.createTrackbar('threshold', 'img_result', 0, 255, self.nothing)
        cv2.setTrackbarPos('threshold', 'img_result', 30)

        while (True):
            self.img_color = cv2.imread(filesource)
            height, width = self.img_color.shape[:2]
            self.img_color = cv2.resize(self.img_color, (width, height),
                                       interpolation=cv2.INTER_AREA)

            # ?????? ????????? HSV ???????????? ???????????????.
            img_hsv = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2HSV)

            # ?????? ????????? HSV ??????????????? ???????????? ???????????????.
            img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
            img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
            img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
            img_mask = img_mask1 | img_mask2 | img_mask3

            kernel = np.ones((11, 11), np.uint8)
            img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
            img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

            # ????????? ???????????? ?????? ??????????????? ???????????? ???????????? ?????? ????????? ???????????????.
            img_result = cv2.bitwise_and(self.img_color, self.img_color, mask=img_mask)

            numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(
                img_mask)

            for idx, centroid in enumerate(centroids):
                if stats[idx][0] == 0 and stats[idx][1] == 0:
                    continue

                if np.any(np.isnan(centroid)):
                    continue

                x, y, width, height, area = stats[idx]
                centerX, centerY = int(centroid[0]), int(centroid[1])
                # print(centerX, centerY)

                if area > 50:
                    cv2.circle(self.img_color, (centerX, centerY), 10, (0, 0, 255), 10)
                    cv2.rectangle(self.img_color, (x, y), (x + width, y + height),
                                  (0, 0, 255))

            cv2.imshow('img_color', self.img_color)
            cv2.imshow('img_mask', img_mask)
            cv2.imshow('img_result', img_result)

            # ESC ???????????? ??????
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


    # def selectcolor(self):
    #     color = QColorDialog.getColor()
    #     if color.isValid():
    #         color = ImageColor.getcolor(color.name(), "HSV")
    #         print(color)
    #         # print(color.name())

# ????????? ????????????
    def video(self, image):
        self.videoLb.setPixmap(QPixmap.fromImage(image).scaled(self.videoLb.size(), Qt.KeepAspectRatio))

    def captureslot(self):
        self.th.ndimg.connect(self.capture)

    def capture(self, image):
        print(type(image))
        # print(image.shape)
        cv2.imwrite("./capture/cap01" + ".png", image)
        self.th.ndimg.disconnect(self.capture)


    def detectstart(self):
        global source
        if self.lbox.currentItem() == None:
            QMessageBox.warning(self, '??????', '????????? ????????? ????????? ????????????.  ')
        else:
            source = self.lbox.currentItem().text()
            if source == '' or None:
                QMessageBox.warning(self, '??????', '????????? ????????? ????????? ???????????? ???????????????.  ')
            else:
                print("Detecting start")
                global writer
                fps = 29.97
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                # writer = cv2.VideoWriter(source + '_detecting.avi', fourcc, fps, (640, 360))
                self.t1name = self.t1namelb.text()
                self.t2name = self.t2namelb.text()

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
                                                label = cat['name'] # ???????????? ????????????????????? ????????? person ??????

                                        ## extract every person
                                        if label == 'person':
                                            # crop them
                                            img_color = image_np[ymin:ymax, xmin:xmax]
                                            cv2.imshow('aaa', img_color)
                                            cv2.waitKey(0)







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
            QMessageBox.warning(self, '??????', '??????????????? ???????????? ??????!  ')
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
            QMessageBox.warning(self, '??????', '????????? ????????? ????????? ????????????.  ')
        else:
            source = self.lbox.currentItem().text()
            if source == '' or None:
                QMessageBox.warning(self, '??????', '????????? ????????? ????????? ???????????? ???????????????.  ')
            else:
                self.th.start()
                self.th.changePixmap.connect(self.video)

    def videostop(self):
        if self.th.isRunning():
            self.th.terminate()
            self.th.changePixmap.disconnect(self.video)
        else:
            QMessageBox.warning(self, '??????', '???????????? ???????????? ????????????.')


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()

