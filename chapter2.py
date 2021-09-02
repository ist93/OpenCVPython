import cv2
import numpy as np

videoFile = cv2.VideoCapture("Resources/theball.mp4")

myPoints = []

myColors = [[23, 24, 159, 179, 255, 255]]


def findColor(resizedImage):
    imgHSV = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2HSV)
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        if x != 0 and y != 0:
            newPoints.append([x, y])
    return newPoints


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y


def drawOnCanvas(myPoints):
    for point in myPoints:
        cv2.putText(imgResult, "Start", (myPoints[0][0], myPoints[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.circle(imgResult, (point[0], point[1]), 10, (255, 0, 0), cv2.FILLED)


while videoFile.isOpened():
    success, img = videoFile.read()
    if success:
        resizedImage = cv2.resize(img, (1024, 900))
        imgResult = resizedImage.copy()
        newPoints = findColor(resizedImage)
        if len(newPoints) != 0:
            for newP in newPoints:
                myPoints.append(newP)
        if len(myPoints) != 0:
            drawOnCanvas(myPoints)
        cv2.imshow("Video", imgResult)
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break
    else:
        cv2.imwrite("Resources/Output/Move.jpg", imgResult)
        break
