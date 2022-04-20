
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

# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# sys.path.append("..")
# PATH_TO_CKPT = './model/frozen_inference_graph.pb'
# PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
#
# NUM_CLASSES = 90
#
# detection_graph = tf.compat.v1.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.compat.v1.GraphDef()
#     with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
#                                                             use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

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


    def count_nonblack_np(self, img):
        """Return the number of pixels in img that are not black.
        img must be a Numpy array with colour values along the last axis.

        """
        return img.any(axis=-1).sum()

    def detect_team(self, image, show=False):
        # define the list of boundaries
        boundaries = [
            ([17, 15, 100], [50, 56, 200]),  # red
            ([25, 146, 190], [96, 174, 250])  # yellow
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
            #         print("ratio is:", ratio)
            if ratio > 0.01 and i == 0:
                return 'red'
            elif ratio > 0.01 and i == 1:
                return 'yellow'

            i += 1

            if show == True:
                cv2.imshow("images", np.hstack([image, output]))
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
        return 'not_sure'

# 비디오 재생관리
    def video(self, image):
        self.videoLb.setPixmap(QPixmap.fromImage(image).scaled(self.videoLb.size(), Qt.KeepAspectRatio))

    def colorfinder(self, color):
        if color == 'Red':
            lower = [17, 15, 100]
            upper = [50, 56, 200]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        else:
            lower = [25, 146, 190]
            upper = [96, 174, 250]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

        return lower, upper

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
                t1name = self.t1namelb.text()
                t2name = self.t2namelb.text()
                t1color = str(self.t1colorbox.currentText())
                t2color = str(self.t2colorbox.currentText())

                t1color = self.colorfinder(t1color)
                t2color = self.colorfinder(t2color)

                print(t1color)
                print(t2color)


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
