import cv2

img = cv2.imread('img_1.jpg')
# print(img)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    cv2.rectangle(img,(200,100),(200+200,100+200),(255,0,0),5)
    cv2.putText(img,"MS Dhoni",(200,100),font,1,(0,0,255),2)
    cv2.imshow('result',img)
    if cv2.waitKey(10) == 27:
        break