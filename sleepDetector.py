import cv2
import dlib
from scipy.spatial import distance
import time
def mouth_aspect_ratio(lips):
    A = distance.euclidean(lips[2],lips[10])
    B = distance.euclidean(lips[4],lips[8])
    C = distance.euclidean(lips[0],lips[6])
    mar = (A+B)/(2.0*C)
    return mar
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        lips=[]

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 1)
        for n in range(49,61):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            lips.append((x, y))
            next_point = n+1
            if n == 60:
                next_point = 49
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 1)


        left_ear =  calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        mouthaspect = mouth_aspect_ratio(lips)
        mouth_thresh=round(mouthaspect,2)
        EAR = (left_ear+right_ear)/2
        EAR = round(EAR, 2)

        if mouth_thresh>0.79:
            cv2.putText(frame, "Yawn", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 51, 51), 4)
            print("yawn")
        print(mouth_thresh)                

        if EAR < 0.20:

            time.sleep(2)
            cv2.putText(frame, "DROWSY", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 51, 51), 4)
            cv2.putText(frame, "Are you Sleepy?", (50, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("Drowsy")
        print(EAR)
       


    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
