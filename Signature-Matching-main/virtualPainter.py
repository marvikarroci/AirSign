import cv2
import numpy as np
import os
import time
import handTrackingModule as htm


#########################
brushThickness = 40
#########################


def airsigner():
    retry_button = cv2.imread("images/retry.jpg")
    retry_button = cv2.resize(retry_button, (128, 128))

    check_button = cv2.imread("images/check.jpg")
    check_button = cv2.resize(check_button, (128, 128))

    drawColor = (0, 255, 0)

    xp, yp = 0, 0

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = htm.handDetector(detectionCon=0.85)

    while True:
        # 1. Import image
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img[0:128, 1152:1280] = retry_button
        img[0:128, 0:128] = check_button

        # 2. Find Hand Landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        # if there is no hand being recorded, just display the current image
        if lmList is None:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            img[0:128, 1152:1280] = retry_button
            img[0:128, 0:128] = check_button

        else:
            # tip of index and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Moving mode: Cursor is moving, but not drawing
            if fingers[1] == fingers[2] == 1:
                drawColor = (140, 255, 140)
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                xp, yp = x1, y1

            # Drawing mode : Only Index finger is up
            if fingers[1] == 1 and fingers[2] == 0:
                drawColor = (0, 255, 0)
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                # the cursor was initialized at (0,0) and needs to be updated to current location
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                else:
                    imgCanvas = cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    img = cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1

            # user wants to use retry-button
            if x1 > 1152 and y1 < 128:
                # reset canvas
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                # reset video capture
                success, img = cap.read()
                img = cv2.flip(img, 1)

            # user wants to save image and exit
            if x1 < 128 and y1 < 128:
                break


        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgThresh = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
        imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgThresh)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.imshow('virtualPainter', img)
        cv2.waitKey(1)

    imgGrayCanvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgThresh = cv2.threshold(imgGrayCanvas, 10, 255, cv2.THRESH_BINARY_INV)

    return imgThresh