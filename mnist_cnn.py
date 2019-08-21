import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import save_model
# Размер изображения
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)


def LoadDataset():
    # Загружаем данные
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Преобразование размерности изображений
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    # Нормализация данных
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Преобразуем метки в категории
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test


# Создаем последовательную модель
def Model():
    model = Sequential()

    model.add(Conv2D(75, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(100, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Компилируем модель
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())

    return model


def Train(model, X_train, Y_train):
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=2)

    # Обучаем сеть
    history = model.fit(X_train, Y_train, batch_size=200, epochs=8, validation_split=0.2, verbose=2,
                        callbacks=[early_stopping_callback])

    return model, history


def Prediction(model, x):
    x = x.reshape(1, img_rows, img_cols, 1)
    rez = model.predict(x)

    return rez


model = Model()
X_train, Y_train, X_test, Y_test = LoadDataset()

model, _ = Train(model, X_train, Y_train)
# save_model(model, "./11.h5")
# print(model.get_weights())
a = np.arange(10, 25)
np.random.shuffle(a)
for i in a:
    img = X_test[i]
    cv2.imshow("1231", img)
    print(Prediction(model, img))
    cv2.waitKey(0)


# Оцениваем качество обучения сети на тестовых данных
# scores = model.evaluate(X_test, Y_test, verbose=0)
# print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
