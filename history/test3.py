import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")

classes = []
with open("../yolov3.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread("./imgs/thumb1630.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# outs 는 탐지결과, 탐지된 개체, 해당 위치 및 탐지에 대한 신뢰도에 대한 모든 정보를 포함
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.3
nms_threshold = 0.4
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
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# draw bounding box
font = cv2.FONT_ITALIC
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 1, color, 3)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ### 3. color detection

def get_boxes_coordinate(boxes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    boxes = [boxes[i] for i in indices]
    return [[x, y, x+w, y+h] for [x, y, w, h] in boxes]

box_fin = get_boxes_coordinate(boxes)

def get_crop_img(img, boxes):
    return [img[round(y1):round(y2), round(x1):round(x2)] for [x1, y1, x2, y2] in boxes]

imgs = get_crop_img(img, box_fin) # 사진에서 사람 한명 자르기

def detect_red(img):
    red_range = ([10, 10, 70], [80, 80, 255])
    lower = np.array(red_range[0], dtype = "uint8")
    upper = np.array(red_range[1], dtype = "uint8")
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(cvt, lowerb = lower, upperb = upper)
    result = cv2.bitwise_and(cvt, cvt, mask = mask)
    return result

cv2.imshow("imgs[0]", imgs[0]) # 사진에서 사람 한명 자른거 표시
cv2.waitKey(0)
# plt.imshow(imgs[0])

result = detect_red(imgs[0])
result = np.hstack([imgs[0], result])

cv2.imshow("result", result) # 잘라온 사람 검은색으로 대비 표시
cv2.waitKey(0)
# plt.imshow(result)
print('proportion of red : {}'.format(round(len(result[result!=0])/(result.shape[0]*result.shape[1]*3),2)))