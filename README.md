# ✔ 프로젝트 설명
축구경기 영상 속 선수들을 yolov3를 이용해 찾아내고 찾아낸 선수의 유니폼 히스토그램 값을 가져와서 비교해 팀을 구분합니다.

## ✔ 프로젝트에 사용한 툴
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white)  ![Qt](https://img.shields.io/badge/Qt-41CD52?style=flat&logo=Qt&logoColor=white) ![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=flat&logo=YOLO&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white)

# ✔ 프로젝트 정보
##### 설치방법(Getting Started / Installation)

먼저

pip install -r requirements.txt

필요한 라이브러리들을 설치해줍니다.

그리고

clone한 폴더에 yolov3.cfg, yolov3.weight를 다운로드 해줍니다.

[yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

[yolov3.weights](https://pjreddie.com/darknet/yolo/)

다운로드 후,

![image](https://user-images.githubusercontent.com/95533644/168948586-0dbf0409-db2a-4674-9826-6219d20ce16e.png)

yolov3.cfg 파일의 8, 9번 줄에 있는 width, height를 위 사진과 같이 수정해줍니다

# ✔ 프로젝트 결과물
1. 원하는 축구경기 영상 링크를 가져온 뒤 Videosave 버튼을 눌러 영상을 저장해줍니다. 
![image](https://user-images.githubusercontent.com/95533644/168949760-2eea33fd-6c6a-4489-b987-aa07f0332a95.png)
2. Add 버튼을 눌러 저장된 영상을 가져옵니다.
![image](https://user-images.githubusercontent.com/95533644/169423200-cc2aafad-2ac1-400b-9114-6ee2110e66bd.png)
3. Crop Person 버튼으로 영상에 있는 선수들을 Detecting해 이미지로 저장합니다.
4. TEAM1, TEAM2에 있는 setting 버튼을 눌러 팀명, Crop Person으로 잘라낸 선수들의 사진을 고르고 Save 해줍니다.
![image](https://user-images.githubusercontent.com/95533644/169423311-28e251ca-739c-4ad7-b4ac-ba14fe650bdc.png)
![image](https://user-images.githubusercontent.com/95533644/169423537-4995620e-e3fd-4527-b493-39a433d7040c.png)
5. Detecting 버튼을 눌러 녹화해줍니다.
6. 녹화된 영상을 Open으로 불러와 ▶ 버튼으로 재생해봅니다.
