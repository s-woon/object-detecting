#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import urllib
from pytube import YouTube
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# ### 1. Youtube download
# ### https://niceman.tistory.com/92

# url1 = 'https://www.youtube.com/watch?v=gIxu6YR4agk'
# url2 = 'https://www.youtube.com/watch?v=P3OEvX6MHYw'
# url3 = 'https://www.youtube.com/watch?v=_DzsB6gmel4'
# url4 = 'https://www.youtube.com/watch?v=lh4o0I6q75I'
# url5 = 'https://www.youtube.com/watch?v=SviRpR2GQ3s'
# url6 = 'https://www.youtube.com/watch?v=4v48ABmJ9Ts'
# url7 = 'https://www.youtube.com/watch?v=awhQOwtc-c4'
#
# def youtube_downloader(url, path):
#     yt = YouTube(url)
#     vid = yt.streams.all()
#     vid[1].download(path)
#
# youtube_downloader(url1, './videos')
# youtube_downloader(url2, './videos')
# youtube_downloader(url3, './videos')
# youtube_downloader(url4, './videos')
# youtube_downloader(url5, './videos')
# youtube_downloader(url6, './videos')
# youtube_downloader(url7, './videos')

# ### 2. YOLOv3

# get_ipython().system('wget https://pjreddie.com/media/files/yolov3.weights')

import cv2
weight = './yolov3.weights'
cfg = './yolov3.cfg'
net = cv2.dnn.readNet(weight, cfg)

classes = None
with open('./yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

import os
img_list = os.listdir('./imgs/')
img = cv2.imread('./imgs/' + img_list[10])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

import numpy as np
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

img_rescale = cv2.resize(img, (224, 224))
scale = 0.00392
blob = cv2.dnn.blobFromImage(img_rescale, scale, (416,416), (0,0,0), True, crop=False)

len(classes)

def get_output_layers(net):   
    layer_names = net.getLayerNames()    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

net.setInput(blob)
outs = net.forward(get_output_layers(net))
# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.3
nms_threshold = 0.4

Width = img.shape[1]
Height = img.shape[0]

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        if class_id == 0:         
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# go through the detections remaining
# after nms and draw bounding box
for i in indices:
    i = i
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

cv2.imshow("aa", img)
cv2.waitKey(0)
# plt.imshow(img)

# ### 3. color detection

def get_boxes_coordinate(boxes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    boxes = [boxes[i] for i in indices]
    return [[x, y, x+w, y+h] for [x, y, w, h] in boxes]

box_fin = get_boxes_coordinate(boxes)

def get_crop_img(img, boxes):
    return [img[round(y1):round(y2), round(x1):round(x2)] for [x1, y1, x2, y2] in boxes]

imgs = get_crop_img(img, box_fin)

def detect_red(img):
    red_range = ([10, 10, 70], [80, 80, 255])
    lower = np.array(red_range[0], dtype = "uint8")
    upper = np.array(red_range[1], dtype = "uint8")
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(cvt, lowerb = lower, upperb = upper)
    result = cv2.bitwise_and(cvt, cvt, mask = mask)
    return result

plt.imshow(imgs[0])

result = detect_red(imgs[0])
result = np.hstack([imgs[0], result])
plt.imshow(result)
print('proportion of red : {}'.format(round(len(result[result!=0])/(result.shape[0]*result.shape[1]*3),2)))


# ### 4. person image crop

PATH = './person_imgs'
# os.mkdir(PATH)

import cv2
weight = './yolov3.weights'
cfg = './yolov3.cfg'
net = cv2.dnn.readNet(weight, cfg)

classes = None
with open('./yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def getFrame(sec, video_path):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = video.read()    
    return hasFrames, image

def get_output_layers(net):   
    layer_names = net.getLayerNames()    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_boxes_coordinate(boxes, confidences, conf_t, nms_t):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_t, nms_t)
    boxes = [boxes[i] for i in indices]
    return [[x, y, x+w, y+h] for [x, y, w, h] in boxes]

def get_crop_img(img, boxes):
            return [img[round(y1):round(y2), round(x1):round(x2)] for [x1, y1, x2, y2] in boxes]


def get_person_imgs(img, net, conf_t, nms_t, path):
    Width = img.shape[1]
    Height = img.shape[0]
    boxes = []
    confidences = []
    img_rescale = cv2.resize(img, (224, 224))
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(img_rescale, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
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
        
    box_fin = get_boxes_coordinate(boxes, confidences, conf_t, nms_t)
    imgs = get_crop_img(img, box_fin)
    if len(imgs) != 0:
        for i in range(len(imgs)):
            try: 
                cv2.imwrite(path+str(i)+'.jpg', imgs[i])
            except:
                pass

# for i in range(150):
#     success, img = getFrame(60 + i*2, './videos/[2021 PL 10R] S햄튼 vs 맨유 HL.mp4')
#     if success:
#         path = './person_imgs/' + 'video1_frame' + '_' + str(i) + '_'
#         get_person_imgs(img, net, 0.3, 0.4, path)
#
# for i in range(150):
#     success, img = getFrame(60 + i*2, './videos/[2021 PL 27R] 맨시티 vs 맨유 HL.mp4')
#     if success:
#         path = './person_imgs/' + 'video2_frame' + '_' + str(i) + '_'
#         get_person_imgs(img, net, 0.3, 0.4, path)
#
# for i in range(150):
#     success, img = getFrame(60 + i*2, './videos/[2021 PL 35R] 아스톤 빌라 vs 맨유 HL.mp4')
#     if success:
#         path = './person_imgs/' + 'video3_frame' + '_' + str(i) + '_'
#         get_person_imgs(img, net, 0.3, 0.4, path)
#
# for i in range(150):
#     success, img = getFrame(60 + i*2, './videos/[2021 PL 31R] 토트넘 vs 맨유 HL.mp4')
#     if success:
#         path = './person_imgs/' + 'video4_frame' + '_' + str(i) + '_'
#         get_person_imgs(img, net, 0.65, 0.3, path)
#
# for i in range(150):
#     success, img = getFrame(60 + i*2, './videos/[2021 UEL 4강 2차] AS로마 vs 맨유 HL.mp4')
#     if success:
#         path = './person_imgs/' + 'video5_frame' + '_' + str(i) + '_'
#         get_person_imgs(img, net, 0.65, 0.3, path)
#
# for i in range(150):
#     success, img = getFrame(60 + i*2, './videos/[2021 UEL] 소시에다드 vs 맨유 HL.mp4')
#     if success:
#         path = './person_imgs/' + 'video6_frame' + '_' + str(i) + '_'
#         get_person_imgs(img, net, 0.65, 0.3, path)

# get_ipython().system('tar chvfz notebook.tar.gz ./person_imgs*')

len(os.listdir('./person_imgs'))

train_valid_test_split('./player_train', 0.9, 0.1)


# ### 4. MobileNet v2 fine-tuning

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

mobilenet = MobileNetV2(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet') 

mobilenet.trainable = True
for i in range(len(mobilenet.layers)):
    mobilenet.layers[i].trainable = False

model = Sequential()
model.add(mobilenet)
model.add(MaxPooling2D(3))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(6, activation = 'softmax'))

model.summary()

def load_dataset(path):
    data = []
    img_list = os.listdir(path)
    try:
        img_list.remove('.ipynb_checkpoints')
    except:
        pass
    
    for i in range(len(img_list)):
        img = load_img(path + img_list[i], target_size=(128, 128))
        img = img_to_array(img)
        img = preprocess_input(img)
        data.append(img)
    return data

PATH = './player_train/'
class_list = os.listdir(PATH)
class_list.remove('.ipynb_checkpoints')
labels =[]
data = np.array([])
for i in range(len(class_list)):
    data_folder = load_dataset(PATH + class_list[i] + '/')
    data = np.append(data, data_folder)
    label = len(data_folder)*[class_list[i]]
    labels.extend(label)

data = data.reshape(-1, 128, 128, 3)
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.2, stratify=labels)

print(x_train.shape, x_test.shape)

generator = ImageDataGenerator(horizontal_flip=True, zoom_range=0.1, fill_mode="nearest")

INIT_LR = 1e-4
EPOCHS = 30
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
ckp = ModelCheckpoint('./best_model4.h5', save_best_only=True, monitor = 'val_accuracy')

history = model.fit(
        generator.flow(x_train, y_train, batch_size=32),
        steps_per_epoch = len(x_train) // 32,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        validation_steps=len(x_test)//32,
        callbacks = [ckp])

fig, ax = plt.subplots(1,2, figsize = (12, 5))
ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])
ax[0].set_title('Loss')
ax[1].plot(history.history['accuracy'])
ax[1].plot(history.history['val_accuracy'])
ax[1].set_title('Accuracy')
plt.legend(['train', 'validation'])

best_model = load_model('./best_model3.h5')

best_model.evaluate(x_test, y_test)

def get_boxes_coordinate(boxes, confidence, conf_t, nms_t):
    indices = cv2.dnn.NMSBoxes(boxes, confidence, conf_t, nms_t)
    boxes = [boxes[i[0]] for i in indices]
    return [[x, y, x+w, y+h] for [x, y, w, h] in boxes]
    
def get_output_layers(net):   
    layer_names = net.getLayerNames()    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
def get_crop_img(img, boxes):
    return [img[round(y1):round(y2), round(x1):round(x2)] for [x1, y1, x2, y2] in boxes]

def draw_bounding_box(img, class_str, box, color):
    cv2.rectangle(img, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), color, 2)
    cv2.putText(img, class_str, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect_MU(img, net, pred_t_player, pred_t_referee ,classifier, conf_t, nms_t, mu_color = 'home', referee_color = 'black'):
    img_rescale = cv2.resize(img, (224, 224))
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(img_rescale, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    Width = img.shape[1]
    Height = img.shape[0]
    confidences = []
    boxes = []

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
        
    box_fin = get_boxes_coordinate(boxes, confidences, conf_t, nms_t)
    imgs = get_crop_img(img, box_fin)
    try:
        imgs_arr = np.array([])
        
        for i in range(len(imgs)):
            image = cv2.resize(imgs[i], (128, 128))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_input(image)
        
            image = image.reshape(1, 128, 128, 3)
            imgs_arr = np.append(imgs_arr, image)
        
        imgs_arr = imgs_arr.reshape(-1, 128, 128 ,3)
        classes = classifier.predict(imgs_arr)
        
        ### remove ohter classes
        if mu_color == 'home':
            classes[:, [0,2]] = 0
        elif mu_color == 'away':
            classes[:, [1,2]] = 0
        else:
            classes[:, [0,1]] = 0
        
        if referee_color == 'black':
            classes[:, [4,5]] = 0
        elif referee_color == 'blue':
            classes[:, [3,5]] = 0
        else:
            classes[:, [3,4]] = 0
        
        prob = np.max(classes, axis = 1)
        classes = np.argmax(classes, axis = 1)

        class_list = []
        for i in range(len(classes)):
            if (prob[i] > pred_t_player) & (classes[i] < 3):
                class_list.append('MU')
            elif (prob[i] > pred_t_referee) & (classes[i] >=3):
                class_list.append('referee')
            else:
                class_list.append('other')
        ### drawing bounding boxes
        for i in range(len(class_list)):
            if class_list[i] == 'MU':
                draw_bounding_box(img, class_str = class_list[i], box = box_fin[i], color = [0, 0, 255])
            elif class_list[i] == 'referee':
                draw_bounding_box(img, class_str = class_list[i], box = box_fin[i], color = [150, 150, 0])
            else:
                draw_bounding_box(img, class_str = class_list[i], box = box_fin[i], color = [255, 0, 0])
    except:
        pass
    
    return img

img_list = os.listdir('./imgs/')
img = cv2.imread('./imgs/' + img_list[100])
img = detect_MU(img, net, conf_t = 0.7, nms_t = 0.3, pred_t_player = 0.7, pred_t_referee = 0.1 ,classifier = best_model, mu_color = 'home', referee_color = 'black')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

img_list = os.listdir('./imgs/')
img = cv2.imread('./imgs/' + img_list[221])
img = detect_MU(img, net, conf_t = 0.7, nms_t = 0.3, pred_t_player = 0.8, pred_t_referee = 0.1 ,classifier = best_model, mu_color = 'away', referee_color = 'blue')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

img_list = os.listdir('./imgs/')
img = cv2.imread('./imgs/' + img_list[323])
img = detect_MU(img, net, conf_t = 0.7, nms_t = 0.3, pred_t_player = 0.7,pred_t_referee = 0.7 ,classifier = best_model, mu_color = 'third', referee_color = 'yellow')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)


# ### 5. Write videos

def getFrame(sec, video_path):
    video = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = video.read()
    return hasFrames, image

video1 = './videos/[2021 PL 27R] 맨시티 vs 맨유 HL.mp4'
video2 = './videos/[2021 PL 10R] S햄튼 vs 맨유 HL.mp4'
video3 = './videos/[2021 PL 35R] 아스톤 빌라 vs 맨유 HL.mp4'

getFrame(1200, video1)

def write_video(video_path, save_path, length,start_sec, conf_t,pred_t_player, pred_t_referee, mu_color = 'home', referee_color = 'black'):
    cap = cv2.VideoCapture(video_path) 
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (1280,720))
    sec = start_sec
    success, img = getFrame(sec, video_path)
    count = 0
    while success:
        count += 1
        sec += 1/20
        sec = round(sec, 2)
        success, img = getFrame(sec, video_path)
        result_img = detect_MU(img, net, conf_t = conf_t, nms_t = 0.3, pred_t_player = pred_t_player, pred_t_referee = pred_t_referee ,classifier = best_model, mu_color = mu_color, referee_color = referee_color)      
        out.write(result_img)
        
        if (1/20*count) % 1 == 0:
            print(1/20*count)
        
        if 1/20*count == length:
            break
        
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

import sys
import time

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

write_video(video1, './MU_home.avi', 
            start_sec = 70, conf_t = 0.7, 
            pred_t_player =0.8,pred_t_referee =0.1, 
            mu_color = 'home', referee_color = 'black', 
            length = 20)

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

write_video(video2, './MU_away2.avi',
            start_sec = 70, conf_t = 0.7 ,
            pred_t_player =0.8, pred_t_referee =0.1, 
            mu_color = 'away', referee_color = 'blue', 
            length = 20)

write_video(video3, './MU_third.avi',
            start_sec = 90,conf_t = 0.7, 
            pred_t_player =0.7, pred_t_referee =0.1, 
            mu_color = 'third', referee_color = 'yellow', 
            length = 20)

write_video('./videos/[2021 PL 30R] 맨유 vs 브라이튼 HL.mp4', './MU_home_test.avi', 
            start_sec = 70, conf_t = 0.7, 
            pred_t_player =0.8,pred_t_referee =0.1, 
            mu_color = 'home', referee_color = 'black', 
            length = 20)

