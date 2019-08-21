import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import save_model, load_model
import tensorflow as tf
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
    print(model.summary())

    return model


def CreateDataset(path):
    fileP = os.listdir(path + "positive/")
    fileN = os.listdir(path + "negative/")
    size = len(fileP) + len(fileN)
    X = np.zeros([size, img_rows, img_cols, 1])
    Y = np.zeros([size, 2])
    i = 0
    for name in fileP:
        img = cv2.imread(path + "positive/" + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        img = img.astype(np.float)
        img /= 255
        X[i] = img
        Y[i] = [1, 0]
        i += 1

    for name in fileN:
        img = cv2.imread(path + "negative/" + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        img = img.astype(np.float)
        img /= 255
        X[i] = img
        Y[i] = [0, 1]
        i += 1

    inx = np.arange(0, size)
    np.random.shuffle(inx)
    X = X[inx]
    Y = Y[inx]

    return X, Y


X, Y = CreateDataset("./")
model = Model()
# model.fit(X, Y, validation_split=0.1,batch_size=68, epochs=40, verbose=1)

# model.save("./weh.h5")
model.load_weights("./weh.h5")
# save_model(model, "./11.h5")

