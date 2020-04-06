#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:41:12 2020

@author: arjun
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

# 109.37907259306508 141.5884302305364 198.81072981285115
# final strd-  15.300070968207018 19.82259908014858 22.924406388112377
x=list(range(0, 256))

meang1=181.9749070483697 
meang2=227.42128696543588
meang3= 249.1158501022877

stdg1=23.34595736879849 
stdg2=14.994679576212908 
stdg3=6.110445108519966

mg1=np.array([meang1])
mg2=np.array([meang2])
mg3=np.array([meang3])
sg1=np.array([stdg1])
sg2=np.array([stdg2])
sg3=np.array([stdg3])


g2=gaussian(x,np.array([216.02]), np.array([16.884]))
g3=gaussian(x,np.array([242.35]), np.array([6.70]))
r2=gaussian(x, np.array([141.58]),np.array([19.82]) )
r1=gaussian(x, np.array([109.37]),np.array([15.30]) )
b1=gaussian(x, np.array([118.98]),np.array([11.30]) )
b3=gaussian(x, np.array([132.60]),np.array([16.84]) )

#

path=os.getcwd()+"/"+"detectbuoy.avi"
c=cv2.VideoCapture(path)
    
while (True):
    ret,image=c.read()
    b,g,r=cv2.split(image)
    if ret == True:
        img_out3=np.zeros(g.shape, dtype = np.uint8)      
        for index, v in np.ndenumerate(g):
                 if (g3[v]>0.054 or g2[v]>0.021) and (r2[r[index]] >0.018 or r1[r[index]] >0.025) and(b1[b[index]]>0.024 or b3[b[index]]>0.022 ):
                            img_out3[index]=255
                 else:
                            img_out3[index]=0  
        ret, threshold3 = cv2.threshold(img_out3, 240, 255, cv2.THRESH_BINARY)
        kernel3 = np.ones((2,2),np.uint8)
        dilation3 = cv2.dilate(threshold3,kernel3,iterations =8)
        contours3, _= cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        for contour in contours3:
            if cv2.contourArea(contour) > 40:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                print(radius)
                if radius > 12 and radius < 17:
                    cv2.circle(image,center,radius,(0,0,255),2)
                    
        cv2.imshow("Threshold",dilation3)
        cv2.imshow('Green Ball Segmentation', image)
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break      # wait for ESC key to exit
    else:
        break
        
cv2.release()
cv2.destroyAllWindows()   





