from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from elice_utils import EliceUtils
elice_utils = EliceUtils()

    
    
def data_print():
    
    img = cv2.imread("data/numbers.jpg")
    
    print("원본 이미지를 출력합니다.")
    plt.figure(figsize=(15,12))
    plt.imshow(img);
    
    plt.savefig("result2.png")
    elice_utils.send_image("result2.png")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_th = cv2.threshold(img_blur, 155, 250, cv2.THRESH_BINARY_INV)[1]
    
    contours, hierachy= cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rects = [cv2.boundingRect(each) for each in contours]
    
    tmp = [w*h for (x,y,w,h) in rects]
    tmp.sort()
    
    rects = [(x,y,w,h) for (x,y,w,h) in rects if ((w*h>1000)and(w*h<500000))]
    
    print("\n이미지를 분할할 영역을 표시합니다.")
    for rect in rects:
    # Draw the rectangles
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 
    plt.clf()
    plt.figure(figsize=(15,12))
    plt.imshow(img);
    plt.savefig("result3.png")
    elice_utils.send_image("result3.png")
    
    seg_img = []

    margin_pixel = 50

    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
        if ((w*h>1000)and(w*h<500000)): 

            # Cropping the text block for giving input to OCR 
            cropped = img.copy()[y - margin_pixel:y + h + margin_pixel, x - margin_pixel:x + w + margin_pixel] 
            seg_img.append(cropped)

            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 5)

    print("\n분할 된 이미지를 출력합니다.")
    for i in range(len(seg_img)):
        plt.clf()
        plt.imshow(seg_img[i]);
        plt.savefig("result4.png")
        elice_utils.send_image("result4.png")
    
    
    re_seg_img = []

    for i in range(len(seg_img)):
        re_seg_img.append(cv2.resize(seg_img[i], (28,28), interpolation=cv2.INTER_AREA))
    
    gray = cv2.cvtColor(re_seg_img[0], cv2.COLOR_BGR2GRAY)
    