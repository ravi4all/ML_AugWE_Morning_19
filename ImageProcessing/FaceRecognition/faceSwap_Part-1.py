import cv2
import numpy as np

data = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)

swapImg = cv2.imread('dhoni.jpg')
while True:
    flag, img = capture.read()
    if flag:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = data.detectMultiScale(gray)
        swapFace = data.detectMultiScale(swapImg,1.3)
        sx,sy,sw,sh = swapFace[0]
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 200), 4)

            face = img[y:y+h,x:x+w,:]
            swapFace = swapImg[sy:sy+sh,sx:sx+sw,:]
            swapFace = cv2.resize(swapFace,(face.shape[0],face.shape[1]))
            img[y:y + h, x:x + w, :] = swapFace
            
        cv2.imshow('result',img)
        #cv2.imshow('result',swapImg)
        if cv2.waitKey(10) == 27:
            break
    else:
        print("Camera not working")

capture.release()
cv2.destroyAllWindows()
