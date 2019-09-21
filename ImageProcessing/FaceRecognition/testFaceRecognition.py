import numpy as np
import pickle
import cv2

font = cv2.FONT_HERSHEY_COMPLEX

dataset = cv2.CascadeClassifier('data.xml')
img = cv2.imread('test_1.jpg',cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = dataset.detectMultiScale(gray,1.3)

file = open('weights.pkl','rb')
weights = pickle.load(file)
file.close()
names = {
    0 : "Ravi",
    1 : "Unknown"
}

def prediction(x, w):
    z = np.dot(x, w)
    return 1 / (1 + np.exp(-z))

# cap = cv2.VideoCapture(0)
# while True:
#     _, img = cap.read()
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = dataset.detectMultiScale(gray,1.3)
#     for x,y,w,h in faces:
#         face = gray[y:y+h,x:x+w]
#         face = cv2.resize(face,(50,50))
#         pred = prediction(face.flatten(),weights)
#         pred = round(pred)
#         name = names[int(pred)]
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),5)
#         cv2.putText(img,name,(x,y),font,1,(0,255,255),2)
#     cv2.imshow('result',img)
#     if cv2.waitKey(2) == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

for x,y,w,h in faces:
    face = gray[y:y+h,x:x+w]
    face = cv2.resize(face,(50,50))
    pred = prediction(face.flatten(),weights)
    pred = round(pred)
    name = names[int(pred)]
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),5)
    cv2.putText(img,name,(x,y),font,1,(0,255,255),2)

cv2.imwrite('result.jpg',img)