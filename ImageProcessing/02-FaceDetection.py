import cv2

dataset = cv2.CascadeClassifier('data.xml')

# img = cv2.imread('bahubali.jpg',cv2.COLOR_BGR2GRAY)
# img = cv2.imread('image_1.jpg',cv2.COLOR_BGR2GRAY)
img = cv2.imread('img_1.jpg',cv2.COLOR_BGR2GRAY)
faces = dataset.detectMultiScale(img,1.28)
# print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,200),4)

cv2.imwrite('result_3.jpg',img)