import numpy as np
import dlib
import cv2
import time
import winsound

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

frame_width = 640
frame_height = 480

title_name = 'Drowsiness Detection'

face_cascade_name = './haarcascade_frontalface_alt.xml' #-- 본인 환경에 맞게 변경할 것
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

predictor_file = './shape_predictor_68_face_landmarks.dat' #-- 본인 환경에 맞게 변경할 것
predictor = dlib.shape_predictor(predictor_file)

status = 'Awake'
number_closed = 0
min_EAR = 0.25
closed_limit = 10 #-- 눈 감김이 10번 이상일 경우 졸음으로 간주
show_frame = None
sign = None
color = None

def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)
    
def detectAndDisplay(image):
    global number_closed
    global color
    global show_frame
    global sign

    image = cv2.resize(image, (frame_width, frame_height))
    show_frame = image
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
        show_parts = points[EYES]
        right_eye_EAR = getEAR(points[RIGHT_EYE])
        left_eye_EAR = getEAR(points[LEFT_EYE])
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2 

        right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
        left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")

        cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0,0], right_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0,0], left_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for (i, point) in enumerate(show_parts):
            x = point[0,0]
            y = point[0,1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
            
        if mean_eye_EAR > min_EAR:
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 1
            if( number_closed<0 ):
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'sleep'
            number_closed = number_closed + 1
                     
        sign = 'sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)

        # 졸음 확정시 알람 설정
        if( number_closed > closed_limit ):
            show_frame = frame_gray
            #winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME) #-- 본인 환경에 맞게 변경할 것
        
    cv2.putText(show_frame, status , (x-w,y-h), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)
    cv2.putText(show_frame, sign , (10,frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow(title_name, show_frame)
    
def checkUserEye(image):
    global color
    global show_frame
    global min_EAR
    global right_eye_EAR
    global left_eye_EAR

    image = cv2.resize(image, (frame_width, frame_height))
    show_frame = image
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
        
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
        right_eye_EAR = getEAR(points[RIGHT_EYE])
        left_eye_EAR = getEAR(points[LEFT_EYE])

    min_EAR = (right_eye_EAR + left_eye_EAR) /2
       
                
           

cap = cv2.VideoCapture(0)
time.sleep(2.0)
if not cap.isOpened:
    print('Could not open video')
    exit(0)

input_start_time = time.time()
input_duration = 5  # 사용자 입력을 받을 시간 (초)

print("min_EAR 값을 조정합니다. 다음 5초 동안 졸린 눈을 해주세요")

# 사용자 입력을 받아 min_EAR 조정
while time.time() - input_start_time < input_duration:
    ret, frame = cap.read()

    if frame is None:
        print('프레임을 읽을 수 없습니다.')
        cap.release()
        break

    # 눈 정보를 얻고 표시하는 함수 호출
    checkUserEye(frame)
    cv2.imshow(title_name, frame)

# 입력 받은 값을 토대로 min_EAR 설정
print(f"min_EAR을 {min_EAR}로 설정했습니다.")


while True:
    ret, frame = cap.read()

    if frame is None:
        print('Could not read frame')
        cap.release()
        break

    detectAndDisplay(frame)
    
    # q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()