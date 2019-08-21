import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import load_model
i

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
oldix, oldiy, oldx, oldy = -1, -1, -1, -1
# roi = (0, 0, 0, 0)
rect = (0, 0, 1, 1)

img_rows, img_cols = 48, 32
input_shape = (img_rows, img_cols, 1)


def Model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,
                     padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    # Компилируем модель
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        sceneCopy = sceneImg.copy()
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        roi = sceneCopy[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        roi1 = cv2.resize(roi, dsize=(32, 48))
        cv2.imshow('mouse input', roi1)
        cv2.imshow('cap', roi)



cam = cv2.VideoCapture("D:/12.mp4")
model = Model()
model.load_weights("./weh.h5")
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    _, img = cam.read()
    cv2.imshow('image', img)
    print("Cap? - s")
    k = cv2.waitKey(0)
    print(k)
    if k == 97:
        roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

        roi = cv2.resize(roi, dsize=(32, 48))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = np.expand_dims(roi, axis=2)
        roi = np.expand_dims(roi, axis=0)
        roi = roi.astype(np.float)
        roi /= 255
        pred = model.predict(roi)
        a = np.argmax(pred)
        if a == 0:
            print("Занято")
        else:
            print("Свободно")

    if k == 115:
        sceneImg = img.copy()
    if k == 113:
        break
