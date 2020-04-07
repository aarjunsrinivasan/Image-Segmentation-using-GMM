#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

x=list(range(0, 256))
gr1=gaussian(x,np.array([234.35]), np.array([2.418]))
gr2=gaussian(x,np.array([219.81]), np.array([25.896]))
b1=gaussian(x, np.array([103.602]),np.array([18.64]) )
b2=gaussian(x, np.array([164.969]),np.array([27.56]) )




path=os.getcwd()+"/"+"detectbuoy.avi"
c=cv2.VideoCapture(path)
    
while (True):
    ret,image=c.read()
    b,g,r=cv2.split(image)
    if ret == True:
        img_out3=np.zeros(g.shape, dtype = np.uint8)      
        for index, v in np.ndenumerate(g):
                 av=int((int(v)+int(r[index]))/2)
                 if ( gr1[av]>0.14) and(b1[b[index]]>0.015 or b2[b[index]]>0.012 ):
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
                if radius > 12:
                    cv2.circle(image,center,radius,(0,0,255),2)
                    
        cv2.imshow("Threshold",dilation3)
        cv2.imshow('Yellow Ball Segmentation', image)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break      # wait for ESC key to exit
    else:
        break
        
cv2.release()
cv2.destroyAllWindows()   





