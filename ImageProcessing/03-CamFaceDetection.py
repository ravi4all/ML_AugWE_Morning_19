import cv2

data = cv2.CascadeClassifier('data.xml')
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('video_1.mp4')
while True:
    flag, img = capture.read()
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    if flag:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = data.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 200), 4)

        cv2.imshow('result',img)
        if cv2.waitKey(10) == 27:
            break
    else:
        print("Camera not working")

capture.release()
cv2.destroyAllWindows()