import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUi
from PyQt5 import QtGui


class Camera(QWidget):
    def __init__(self, parent=None, thread=None):
        super(Camera, self).__init__(parent)
        loadUi('camera.ui', self)
        self.parent = parent
        self.thread = thread
        self.thread.Frame.connect(self.setRGB)

        self.camerabtn.clicked.connect(self.cameraon)

        self.sobelBtn.clicked.connect(self.sobeledgeSlot)
        self.laplacianBtn.clicked.connect(self.laplacianedgeSlot)
        self.cannyBtn.clicked.connect(self.cannyedgeSlot)

    def cameraon(self):
        self.thread.changePixmap.connect(self.setImage)

        self.thread.start()

    def setImage(self, image):
        self.originCam.setPixmap(QPixmap.fromImage(image).scaled(self.originCam.size(), Qt.KeepAspectRatio))

    def setRGB(self, frame):
        R, G, B = cv2.split(frame)
        zeros = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

        imgR = cv2.merge((R, zeros, zeros))
        imgG = cv2.merge((zeros, G, zeros))
        imgB = cv2.merge((zeros, zeros, B))
        imgRG = cv2.merge((R, G, zeros))
        imgRB = cv2.merge((R, zeros, B))
        imgGB = cv2.merge((zeros, G, B))

        h, w, c = frame.shape

        Rimg = QtGui.QImage(imgR.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio)
        Gimg = QtGui.QImage(imgG.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio)
        Bimg = QtGui.QImage(imgB.data, w, h, w*c, QtGui.QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio)
        RGimg = QtGui.QImage(imgRG.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio)
        RBimg = QtGui.QImage(imgRB.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio)
        GBimg = QtGui.QImage(imgGB.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio)
        self.lbR.setPixmap(QPixmap.fromImage(Rimg).scaled(self.lbR.size(), Qt.KeepAspectRatio))
        self.lbG.setPixmap(QPixmap.fromImage(Gimg).scaled(self.lbG.size(), Qt.KeepAspectRatio))
        self.lbB.setPixmap(QPixmap.fromImage(Bimg).scaled(self.lbR.size(), Qt.KeepAspectRatio))
        self.lbRG.setPixmap(QPixmap.fromImage(RGimg).scaled(self.lbRG.size(), Qt.KeepAspectRatio))
        self.lbRB.setPixmap(QPixmap.fromImage(RBimg).scaled(self.lbRB.size(), Qt.KeepAspectRatio))
        self.lbGB.setPixmap(QPixmap.fromImage(GBimg).scaled(self.lbGB.size(), Qt.KeepAspectRatio))

    def sobeledgeSlot(self):
        self.thread.changePixmap.connect(self.sobelImage)

    def sobelImage(self, image):
        ndarry = self.qimg2nparr(image)
        sobel1 = cv2.Sobel(ndarry, cv2.CV_32F, 1, 0, 3)  # x 방향 미분 - 수직마스크
        sobel2 = cv2.Sobel(ndarry, cv2.CV_32F, 0, 1, 3)  # y 방향 미분 - 수평마스크
        grayimgX = cv2.cvtColor(sobel1, cv2.COLOR_BGR2GRAY)
        grayimgY = cv2.cvtColor(sobel2, cv2.COLOR_BGR2GRAY)
        sobel1 = cv2.convertScaleAbs(grayimgX)
        sobel2 = cv2.convertScaleAbs(grayimgY)
        h1, w1 = sobel1.shape
        h2, w2 = sobel2.shape
        qimg1 = QImage(sobel1.data, w1, h1, QImage.Format_Grayscale8)
        self.edgeCam.setPixmap(QPixmap.fromImage(qimg1).scaled(self.edgeCam.size(), Qt.KeepAspectRatio))

    def laplacianedgeSlot(self):
        self.thread.changePixmap.connect(self.laplacianImage)

    def laplacianImage(self, image):
        ndarry = self.qimg2nparr(image)
        grayimg = cv2.cvtColor(ndarry, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(grayimg, cv2.CV_16S, 1)
        laplacian = cv2.convertScaleAbs(laplacian)  # 절대값
        h, w = laplacian.shape
        qimg = QImage(laplacian.data, w, h, QImage.Format_Grayscale8)
        self.edgeCam.setPixmap(QPixmap.fromImage(qimg).scaled(self.edgeCam.size(), Qt.KeepAspectRatio))

    def cannyedgeSlot(self):
        self.thread.changePixmap.connect(self.cannyImage)

    def cannyImage(self, image):
        ndarry = self.qimg2nparr(image)
        canny = cv2.Canny(ndarry, 50, 200)
        h, w = canny.shape
        qimg = QImage(canny.data, w, h, QImage.Format_Grayscale8)
        self.edgeCam.setPixmap(QPixmap.fromImage(qimg).scaled(self.edgeCam.size(), Qt.KeepAspectRatio))

    def qimg2nparr(self, qimg):
        ''' convert rgb qimg -> cv2 bgr image '''
        h, w = qimg.height(), qimg.width()
        ptr = qimg.constBits()
        ptr.setsize(h * w * 3)
        return np.frombuffer(ptr, np.uint8).reshape(h, w, 3)  # Copies the data
