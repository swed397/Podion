import cv2
import numpy as np

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
oldix, oldiy, oldx, oldy = -1, -1, -1, -1
# roi = (0, 0, 0, 0)
rect = (0, 0, 1, 1)


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

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
i = 0
while True:
    _, img = cam.read()
    cv2.imshow('image', img)
    print("Cap? - s")
    k = cv2.waitKey(0)
    print(k)
    if k == 97:
        roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        roi = cv2.resize(roi, dsize=(32, 48))
        cv2.imwrite("./negative/" + str(i) + ".png", roi)
        i += 1
    if k == 115:
        sceneImg = img.copy()
    if k == 113:
        break
