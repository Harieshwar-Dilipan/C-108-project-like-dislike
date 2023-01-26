import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

hands=mp_hands.Hands(min_detection_confidence=0.4,min_tracking_confidence=0.4 )

tipId=[4,8,12,16,20]

def countFingers(image,handLandmarks,handNo=0):
    if handLandmarks:
        landmarks=handLandmarks[handNo].landmark
        #print(landmarks)
        fingers=[]
        for lm_index in tipId:
            fingerTipY=landmarks[lm_index].y
            fingerBottomY=landmarks[lm_index-2].y
            
            if lm_index==4:
                if fingerTipY<fingerBottomY:
                    text='Like'
                if fingerBottomY<fingerTipY:
                    text='dislike'
        
        #totalFingers=fingers.count(1)
        #text=f'Fingers: {totalFingers}'
        cv2.putText(image,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

def drawHandLandMarks(image,handLandmarks):
    if handLandmarks:
        for landMarks in handLandmarks:
            mp_drawing.draw_landmarks(image,landMarks,mp_hands.HAND_CONNECTIONS)

while True:
    ret,image=cap.read()
    image=cv2.flip(image,1)
    results=hands.process(image)
    handLandmarks=results.multi_hand_landmarks
    #print(handLandmarks)
    drawHandLandMarks(image,handLandmarks)
    countFingers(image,handLandmarks)
    cv2.imshow('hi',image)
    if cv2.waitKey(1)==32:
        break

cap.release()
cv2.destroyAllWindows()