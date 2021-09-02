import cv2
import numpy as np

img = cv2.imread("Resources/shapes_google.png")
imgContour = img.copy()
triangleCount = 0
rectangleCount = 0
squareCount = 0
circleCount = 0
ovalCount = 0
rhombusCount = 0
count = 0
shapes = []


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows
        hor_con = [imgBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getCountours(img):
    global circleCount,triangleCount,rectangleCount,ovalCount,squareCount,rhombusCount
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000 and area < 20000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.009 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 3:
                objectType = "Tri"
                triangleCount = triangleCount+ 1
            elif objCor == 4:
                aspRatio = w / float(h)
                p1 = approx[0][0]
                p2 = approx[1][0]
                p3 = approx[-1][0]
                p4 = approx[2][0]

                firstCornerDiff = p1[0] / float(p4[0])
                secondCornerDiff = p2[1] / float(p3[1])

                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                    squareCount = squareCount + 1
                elif (firstCornerDiff > 0.95 and firstCornerDiff < 1.05) and (
                        secondCornerDiff > 0.95 and secondCornerDiff < 1.05):
                    objectType = "Rhombus"
                    rhombusCount = rhombusCount + 1
                else:
                    objectType = "Rectangle"
                    rectangleCount = rectangleCount + 1
            elif objCor > 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    circleCount = circleCount + 1
                    objectType = "Circle"
                else:
                    ovalCount = ovalCount + 1
                    objectType = "Oval"
            else:
                objectType = "None"
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)


imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 3, 0)
imgCanny = cv2.Canny(imgBlur, 200, 200)
kernel = np.ones((5,5))
imgDial = cv2.dilate(imgCanny,kernel,iterations=4)
getCountours(imgDial)

imgBlank = np.zeros_like(img)
imgStack = stackImages(0.6, ([img, imgGray, imgBlur],
                             [imgCanny, imgContour, imgDial]))
count = triangleCount + rectangleCount + squareCount + rhombusCount + circleCount + ovalCount
print("There are " + str(count) + " shapes: ")
if triangleCount != 0:
    print("Triangles: " + str(triangleCount))
if rectangleCount != 0:
    print("Rectangles: " + str(rectangleCount))
if squareCount!=0:
    print("Squares: " + str(squareCount))
if rhombusCount!=0:
    print("Rhombuses: " + str(rhombusCount))
if circleCount!=0:
    print("Circles: " + str(circleCount))
if ovalCount!=0:
    print("Ovals: " + str(ovalCount))
cv2.imshow("Image", imgStack)
cv2.waitKey(0)
