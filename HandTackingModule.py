# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:03:58 2021

@author: Ashita
"""

import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, max_hands= 2, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con= detection_con
        self.tracking_con = tracking_con
        
        self.mpHands= mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, 
                                        self.max_hands,
                                        self.detection_con,
                                        self.tracking_con)
        self.mpDraw = mp.solutions.drawing_utils
        
    def find_hands(self, image, draw = True):
        imgRGb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGb)
    
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handlms, self.mpHands.HAND_CONNECTIONS)
        return image
    
    def find_position(self, image, handNo=0, draw = True):
        
        lmlist =[]
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                print(id, lm)
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 7, (255,0,0), cv2.FILLED)

            
        return lmlist
    
    
    #k = cv2.waitKey(1)
    
    #if k%256==27:
     #   break
    
def main():
    pTime= 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, image = cap.read()
        image = detector.find_hands(image)
        lm_list = detector.find_position(image)
        if len(lm_list)!=0:
            print(lm_list[4])
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime= cTime
        
        cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
        cv2.imshow("Image", image)
        k = cv2.waitKey(1)
    
        if k%256==27:
            break
    
    
    
if __name__=="__main__":
    main()