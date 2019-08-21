import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import utils
from keras.callbacks import EarlyStopping

def LoadDataset():
    # Загружаем данные
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Преобразование размерности изображений
    X_train = X_train.reshape(60000, 784)  # Преобразование из 3хмерного в 2хмерный массив
    X_test = X_test.reshape(10000, 784)

    # Нормализация данных
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Преобразуем метки в категории
    Y_train = utils.to_categorical(y_train, 10)  # Подготовка выходных данных для обучения с учителем
    Y_test = utils.to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test

def Model():
    model = Sequential()#Создания объекта по классу
    model.add(Dense(784, input_dim=784, activation="relu", name="Input"))#Добавление к объекту 1(входного) слоя. FullyConection
    model.add(Dense(10, activation="softmax"))#Добавление выходного слоя

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])#Настройка LossFunction и метрики
    print(model.summary())#Вывод на экран архитектуры

    return model

def Train(model, X_train, Y_train):

    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=2)
    history = model.fit(X_train, Y_train, batch_size=200, epochs=25, validation_split=0.2, verbose=1,
                    callbacks=[early_stopping_callback])#Запуск процесса обучения с настройками

    return model

def Prediction(model, x):
    x = x.reshape(1, 784) #x = x.reshape(1, img_rows, img_cols, 1)
    rez = model.predict(x)

    return rez

model = Model()
X_train, Y_train, X_test, Y_test = LoadDataset()
model = Train(model, X_train, Y_train)

a = np.arange(10, 25)
np.random.shuffle(a)
for i in a:
    img = X_test[i]
    print(Prediction(model, img))
    # img = img.reshape(28, 28, 1)
    cv2.imshow("1231", img)
    cv2.waitKey(0)