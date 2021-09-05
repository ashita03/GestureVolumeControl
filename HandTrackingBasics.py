# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:55:22 2021

@author: Ashita
"""
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands= mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime= 0
cTime = 0

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                print(id, lm)
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                
                if id==0:
                    cv2.circle(image, (cx, cy), 15, (255,0,255), cv2.FILLED)
                    
            mpDraw.draw_landmarks(image, handlms, mpHands.HAND_CONNECTIONS)
            
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime= cTime
    
    cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
            
            
    
    cv2.imshow("Image", image)
    k = cv2.waitKey(1)
    
    if k%256==27:
        break
    
    
    
