import numpy as np
import dlib
import cv2
import time
import winsound
import time
import pandas as pd
from geopy.distance import geodesic

df = pd.read_csv('data.csv', encoding='cp949')

RIGHT_EYE = list(range(36, 42)) #dlib 라이브러리의 오른쪽 눈, 왼쪽 눈 포인트 값
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

frame_width = 640 #캠 화면 크기
frame_height = 480

title_name = 'Drowsiness Detection'

face_cascade_name = './haarcascade_frontalface_alt.xml' #opencv에서 제공하는 데이터
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

predictor_file = './shape_predictor_68_face_landmarks.dat' #dlib에서 지원하는 얼굴에 특징점을 추정하는 데이터
predictor = dlib.shape_predictor(predictor_file)

status = 'Awake'
number_closed = 0
min_EAR = 0.25 # 눈이 감겼음을 판단하는 수치
closed_limit = 10 #-- 눈 감김이 10번 이상일 경우 졸음으로 간주
show_frame = None
sign = None
sign2 = None
color = None
closed_time = 0
reset_time = 5
last_reset_time = time.time()

def clearCount():
    global number_closed
    number_closed = 0

def getEAR(points): #포인트 값들 사이의 벡터값 비율로 눈이 감긴지 체크
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)
    
def detectAndDisplay(image):
    global number_closed
    global color
    global show_frame
    global sign
    global status
    global closed_time
    global sign2
    current_time = 0
    last_reset_time = 0

    image = cv2.resize(image, (frame_width, frame_height))
    show_frame = image
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)

    x, y, w, h = 0, 0, 0, 0
    
    for (x, y, w, h) in faces: #얼굴에 테두리 출력
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
        show_parts = points[EYES]
        right_eye_EAR = getEAR(points[RIGHT_EYE]) #오른쪽 눈 비율
        left_eye_EAR = getEAR(points[LEFT_EYE]) #왼족 눈 비율
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2 #양쪽 눈 평균

        right_eye_center = np.mean(points[RIGHT_EYE], axis=0).astype("int") 
        left_eye_center = np.mean(points[LEFT_EYE], axis=0).astype("int")

        cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0,0], right_eye_center[0,1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0,0], left_eye_center[0,1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for (i, point) in enumerate(show_parts): #눈 주변의 랜드마크 포인트 출력
            x = point[0,0]
            y = point[0,1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
            
        if mean_eye_EAR > min_EAR: #눈을 잘 뜨고있는 경우
            color = (0, 255, 0)
            status = 'Awake'
            if number_closed < 0:
                number_closed = 0
        else: #눈을 감은 경우
            color = (0, 0, 255)
            status = 'sleep'

            current_time = time.time()
            if current_time - last_reset_time > 0.3: #눈을 감은 횟수 계산 (5초안에 10번 감으면 졸리다고 판단)
                number_closed = number_closed + 1
                last_reset_time = current_time

        
        if mean_eye_EAR < min_EAR: #눈을 계속 감고 있을 경우
            closed_time += 1
        else:
            closed_time = 0
                     
        sign = 'sleep count : ' + str(number_closed) + ' / ' + str(closed_limit) #화면에 값 출력
        sign2 = 'sleep time: ' + str(closed_time) + ' / ' + str(20)

        # 졸음 확정시 알람 설정
        if number_closed > closed_limit or closed_time >= 20:
            show_frame = frame_gray
            winsound.PlaySound("./sound.wav", winsound.SND_FILENAME) #-- 본인 환경에 맞게 변경할 것

            target_latitude = 37.494705 #숭실대학교 위치
            target_longitude = 126.959945

            df['거리'] = ((df['위도'] - target_latitude)**2 + (df['경도'] - target_longitude)**2)**0.5

            near = df.loc[df['거리'].idxmin()]
            name = near[['졸음쉼터명']]
            # 두 지점의 위도와 경도를 설정
            location1 = (37.494705, 126.959945)  # 예시로 숭실대학교의 위치
            location2 = (int(near[['위도']]), int(near[['경도']]))

            distance = geodesic(location1, location2).kilometers

            print('가장 가까운 졸음쉼터는 ' + name.to_string(header=False, index=False) + f"졸음쉼터입니다. ({distance:.2f} 킬로미터)")
            
        
    cv2.putText(show_frame, status , (x-w, y-h), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)
    cv2.putText(show_frame, sign , (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(show_frame, sign2, (10, frame_height), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
    cv2.imshow(title_name, show_frame)
    
cap = cv2.VideoCapture(0)
time.sleep(2.0)
if not cap.isOpened:
    print('Could not open video')
    exit(0)

while True:
    ret, frame = cap.read()

    if frame is None:
        print('Could not read frame')
        cap.release()
        break

    detectAndDisplay(frame)

    current_time = time.time()
    if current_time - last_reset_time >= reset_time: #5초마다 초기화
        clearCount()
        last_reset_time = current_time
    
    # q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()