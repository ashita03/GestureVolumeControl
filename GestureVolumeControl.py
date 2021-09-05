# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:44:57 2021

@author: Ashita
"""

import cv2
import numpy as np
import time
import HandTackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4,hCam)

Ptime=0

detector = htm.handDetector(detection_con=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minvol = volRange[0]
maxvol = volRange[1]

vol=0
volBar = 300
volPer=0

while True:
    success, img = cap.read()
    detector.find_hands(img)
    lmlist = detector.find_position(img, draw= False)
    
    if len(lmlist)!=0:
        #print(lmlist[4], lmlist[8])
        
        x1, y1 =lmlist[4][1], lmlist[4][2]
        x2, y2 =lmlist[8][1], lmlist[8][2]
        
        cx,cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.circle(img,(x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img,(x2,y2), 15, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2),(255,0,255),3 )
        cv2.circle(img,(cx,cy), 15, (255,0,255), cv2.FILLED)
        
        length = math.hypot(x2-x1,y2-y1)
        #print(length)
        
        #Hand Range 50 -300. Now convert this to volume range
        #Volume Range -65 -0
        
        vol =np.interp(length, [50,170],[minvol,maxvol])
        volBar =np.interp(length, [50,170], [400, 150])
        volPer =np.interp(length, [50,170], [0, 100])
        
        print(int(length), vol)
        
        volume.SetMasterVolumeLevel(vol, None)
        
        if length<50:
            cv2.circle(img,(cx,cy), 15, (0,255,0), cv2.FILLED)
            
    cv2.rectangle(img, (50,150), (85,400), (0,255,0),3)
    cv2.rectangle(img, (50,int(volBar)), (85,400), (0,255,0),cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        
    Ctime = time.time()
    fps = 1/ (Ctime - Ptime)
    Ptime = Ctime
    
    cv2.putText(img, f'FPS:{int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    
    if k%256==27:
        break