from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from elice_utils import EliceUtils
elice_utils = EliceUtils()

def train():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

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

    # epochs 값 변경
    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)

    # 임의의 5가지 test data의 이미지와 레이블값을 출력하고 예측된 레이블값 출력
    predictions = model.predict(x_test)
    idx_n = [15, 34, 68, 75, 98]

    for i in idx_n:

        img = x_test[i].reshape(28,28)
        plt.imshow(img,cmap="gray")
        plt.show()
        plt.savefig("result1.png")
        elice_utils.send_image("result1.png")

        print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions[i]))
        
    return model
    