
import sys
import cv2
from time import sleep
import tensorflow as tf
from PIL import ImageColor

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QColorDialog
from PyQt5.uic import loadUi
from pytube import YouTube

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.append("..")
PATH_TO_CKPT = './model/frozen_inference_graph.pb'
PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'

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
        loadUi('main4.ui', self)

        self.th = Thread(self)

        self.saveBtn.clicked.connect(self.videosave)
        self.openBtn.clicked.connect(self.videoopen)
        self.deleteBtn.clicked.connect(self.videodelete)
        self.startBtn.clicked.connect(self.videostart)
        self.stopBtn.clicked.connect(self.videostop)
        # self.selectcolorBtn.clicked.connect(self.selectcolor)

        self.detectBtn.clicked.connect(self.detectstart)

        self.actionOpen.triggered.connect(self.videoopen)

    # def selectcolor(self):
    #     color = QColorDialog.getColor()
    #     if color.isValid():
    #         color = ImageColor.getcolor(color.name(), "RGB")
    #         print(color)
    #         # print(color.name())

# 비디오 재생관리
    def video(self, image):
        self.videoLb.setPixmap(QPixmap.fromImage(image).scaled(self.videoLb.size(), Qt.KeepAspectRatio))

    def colorfinder(self, color):
        if color == 'Red':
            lower = [17, 15, 100]
            upper = [50, 56, 200]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        elif color == 'Orange':
            lower = [0, 100, 240]
            upper = [120, 200, 255]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        elif color == 'Yellow':
            lower = [0, 220, 210]
            upper = [160, 255, 255]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        elif color == 'Green':
            lower = [30, 100, 0]
            upper = [200, 235, 200]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        elif color == 'Blue':
            lower = [150, 100, 0]
            upper = [255, 180, 150]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        elif color == 'Black':
            lower = [0, 0, 0]
            upper = [140, 140, 140]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        else: # Violet
            lower = [150, 50, 100]
            upper = [255, 180, 250]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

        return lower, upper

    def count_nonblack_np(self, img):
        """Return the number of pixels in img that are not black.
        img must be a Numpy array with colour values along the last axis.

        """
        return img.any(axis=-1).sum()

    def detect_team(self, lower, upper, image, show=False):
        # define the list of boundaries
        i = 0

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        tot_pix = self.count_nonblack_np(image)
        color_pix = self.count_nonblack_np(output)
        ratio = color_pix / tot_pix
        #         print("ratio is:", ratio)
        if ratio > 0.01 and i == 0:
            return self.t1color
        elif ratio > 0.01 and i == 1:
            return self.t2color

        if show == True:
            cv2.imshow("images", np.hstack([image, output]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
              cv2.destroyAllWindows()

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
                self.t1name = self.t1namelb.text()
                self.t2name = self.t2namelb.text()
                self.t1color = self.t1colorbox.currentText()
                self.t2color = self.t2colorbox.currentText()
                print(self.t1name, self.t1color, self.t2name, self.t2color)
                lower, upper = self.colorfinder(self.t1color)
                print(lower, upper)
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
                                                label = cat['name']

                                        ## extract every person
                                        if label == 'person':
                                            # crop them
                                            crop_img = image_np[ymin:ymax, xmin:xmax]
                                            color = self.detect_team(lower, upper, crop_img)
                                            if color != 'not_sure':
                                                coords = (xmin, ymin)
                                                if color == self.t1color:
                                                    loc[coords] = self.t1name
                                                else:
                                                    loc[coords] = self.t2name

                                ## print color next to the person
                                for key in loc.keys():
                                    text_pos = str(loc[key])
                                    cv2.putText(image_np, text_pos, (key[0], key[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.50, (255, 0, 0), 2)  # Text in black

                            cv2.imshow('image', image_np)

                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                cap.release()
                                break


                # vid = cv2.VideoCapture(source)
                # while vid:
                #     ret, image_np = vid.read()
                #     if ret:
                #         h = image_np.shape[0]
                #         w = image_np.shape[1]
                #
                #     if not ret:
                #         break

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


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
