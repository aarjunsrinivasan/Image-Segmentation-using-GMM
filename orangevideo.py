#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:06:13 2020

@author: arjun
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


x=list(range(0, 256))


g1=gaussian(x,np.array([207.976]), np.array([26.33]))
g2=gaussian(x,np.array([146.236]), np.array([19.748]))
r2=gaussian(x, np.array([226.02]),np.array([25.02]) )
r1=gaussian(x, np.array([248.196]),np.array([5.679]) )
b1=gaussian(x, np.array([93.84]),np.array([16.93]) )
b2=gaussian(x, np.array([136.613]),np.array([38.30]) )





path=os.getcwd()+"/"+"detectbuoy.avi"
c=cv2.VideoCapture(path)
    
while (True):
    ret,image=c.read()
    b,g,r=cv2.split(image)
    if ret == True:
        img_out3=np.zeros(g.shape, dtype = np.uint8)      
        for index, v in np.ndenumerate(g):
                 if ( g1[v]>0.009 or g2[v]>0.012) and (r1[r[index]] >0.052) and(b1[b[index]]>0.012):
                            img_out3[index]=255
                 else:
                            img_out3[index]=0  
        ret, threshold3 = cv2.threshold(img_out3, 240, 255, cv2.THRESH_BINARY)
        kernel3 = np.ones((2,2),np.uint8)
        dilation3 = cv2.dilate(threshold3,kernel3,iterations =8)
        contours3, _= cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours3:
            if cv2.contourArea(contour) > 50:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                print(radius)
                if 11<radius <30:
                    cv2.circle(image,center,radius,(0,0,255),2)
                    
        cv2.imshow("Threshold",dilation3)
        cv2.imshow('Orange Ball Segmentation', image)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break      # wait for ESC key to exit
    else:
        break
        
cv2.release()
cv2.destroyAllWindows()   





