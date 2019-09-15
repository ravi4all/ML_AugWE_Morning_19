import cv2
import numpy as np

data = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)

faceData = []

while True:
    flag, img = capture.read()
    if flag:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = data.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 200), 4)

            face = gray[y:y+h,x:x+w]
            face = cv2.resize(face,(50,50))
            if len(faceData) < 200:
                faceData.append(face)
                print(len(faceData))

        cv2.imshow('result',img)
        if cv2.waitKey(10) == 27 or len(faceData) >= 200:
            break
    else:
        print("Camera not working")

faceData = np.asarray(faceData)
np.save('user_1.npy',faceData)
capture.release()
cv2.destroyAllWindows()
