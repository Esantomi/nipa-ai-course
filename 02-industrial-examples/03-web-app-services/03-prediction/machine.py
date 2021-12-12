from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

import cv2
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")

from elice_utils import EliceUtils
elice_utils = EliceUtils()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
def data_predit():

    # model에 batch normalization layer 추가
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
      #tf.keras.layers.Dense(10, activation='relu')
    ])

    # adam외의 optimizer로 변경
    # sparse_categorical_crossentropy외의 loss로 변경
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    import os

    checkpoint_path = "./cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    model.load_weights(checkpoint_path)


    
    img = cv2.imread("data/numbers.jpg")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_th = cv2.threshold(img_blur, 150, 250, cv2.THRESH_BINARY_INV)[1]
    
    contours, hierachy= cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rects = [cv2.boundingRect(each) for each in contours]
    
    tmp = [w*h for (x,y,w,h) in rects]
    tmp.sort()
    
    rects = [(x,y,w,h) for (x,y,w,h) in rects if ((w*h>1000)and(w*h<500000))]
    
    for rect in rects:
    # Draw the rectangles
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 
    
    
    seg_img = []

    margin_pixel = 50

    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
        if ((w*h>1500)and(w*h<600000)): 

            # Cropping the text block for giving input to OCR 
            cropped = img.copy()[y - margin_pixel:y + h + margin_pixel, x - margin_pixel:x + w + margin_pixel] 
            seg_img.append(cropped)

            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 5)

    re_seg_img = []

    for i in range(len(seg_img)):
        re_seg_img.append(cv2.resize(seg_img[i], (28,28), interpolation=cv2.INTER_AREA))
    
    gray = cv2.cvtColor(re_seg_img[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(len(seg_img)):
    
        gray = cv2.cvtColor(re_seg_img[i], cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(gray, 150, 250, cv2.THRESH_BINARY_INV)[1]
        test = img_binary.reshape(1,28,28) / 255.0

        #print(len(test))

        predictions = model.predict(test)

        img_test = test.reshape(28,28)
        plt.clf()
        
        plt.subplot(121)
        plt.imshow(seg_img[i])
        plt.title('Origin')

        plt.subplot(122)
        plt.imshow(img_test,cmap="gray")
        plt.title('Coverted')
        plt.show()
        
        plt.savefig("result4.png")
        elice_utils.send_image("result4.png")

        #print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions))