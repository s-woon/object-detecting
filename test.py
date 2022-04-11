from bs4 import BeautifulSoup
import urllib
from pytube import YouTube
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

weight = './yolov3.weights'
cfg = './yolov3.cfg'
net = cv2.dnn.readNet(weight, cfg)

classes = None
with open('./yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
cap = cv2.VideoCapture(0)
while True:
    # img = cv2.imread("imgs/thumb1650.jpg")
    ret, img = cap.read()
    if ret:
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
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
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    print('aa')
    cv2.imshow("image", img)
    print('bb')
    if cv2.waitKey(0) != 0:
        break
    print('cc')
















# img_list = os.listdir('./imgs/')
# img = cv2.imread('./imgs/' + img_list[10])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# cv2.imshow("aa", img)
# cv2.waitKey(0)