import cv2
import mediapipe as mp
import os
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,
                              cv2.COLOR_BGR2RGB)  # bcoz this class only accepts RGB images, so converting it into RGB images
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLns in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLns, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lm = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):

                h, w, c = img.shape
                cx, cy = int(ln.x * w), int(ln.y * h)

                lm.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        return lm


def main():
    wCam, hCam = 1080, 720

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    folderpath = "fingers"
    mylist = os.listdir(folderpath)
    # print(mylist)

    overlaylist = []

    for imPath in mylist:
        image = cv2.imread(f'{folderpath}/{imPath}')
        # print(f'{folderpath}/{imPath}')
        overlaylist.append(image)

    ptime = 0

    detector = handDetector(detectionCon=0.75)

    tipIds = [4, 8, 12, 16, 20]

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm = detector.findPosition(img, draw=False)

        if len(lm) != 0:
            fingers = []

            # THUMB
            if lm[tipIds[0]][1] > lm[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # FOUR FINGERS
            for id in range(1, 5):
                if lm[tipIds[id]][2] < lm[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)
            totalFingers = fingers.count(1)
            # print(totalFingers)

            h, w, c = overlaylist[totalFingers - 1].shape
            img[0:h, 0:w] = overlaylist[totalFingers - 1]
            cv2.putText(img, f'{int(totalFingers)}', (0, 310), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 10)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (700, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()




