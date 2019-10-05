# Using K-NN Machine Learning
import numpy as np
import os
import cv2

trainImgArray = np.load('training_data.npy')
# print(trainImgArray.shape)
train_data = trainImgArray.reshape((trainImgArray.shape[0],-1))

labels = os.listdir('dataset/Training/')
labels_dict = {i : labels[i] for i in range(len(labels))}

def readlabelsLength(path):
    labels_length = []
    for root, folder, files in os.walk(path):
        labels_length.append(len(files))
    return labels_length

trainlabelslength = readlabelsLength(path='dataset/Training/')

output_labels = np.zeros((len(trainImgArray),1), dtype=np.int32)

slice_1 = 0
slice_2 = 0
try:
    for j in range(len(trainlabelslength)-1):
        slice_1 += trainlabelslength[j]
        slice_2 += trainlabelslength[j+1]
        output_labels[slice_1:slice_2] = int(j)
except BaseException:
    print("Index out of range")

# Using KNN
def distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum())

def knn(x, train, k=5):
    m = train.shape[0]
    dist = []
    for i in range(m):
        dist.append(distance(x, train[i]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = output_labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]

#test_img = cv2.imread('test_1.jpg')
#test_img = color.rgb2gray(io.imread('test_1.jpg'))
#lab = knn(test_img.flatten(), train_data)
#text = labels[int(lab)]
#print(text)

font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = frame[100:300, 100:300,:]
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
        obj = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fruit = cv2.resize(obj, (100, 100))
        # print(fruit.shape)
        lab = knn(fruit.flatten(), train_data)
        text = labels[int(lab)]
        cv2.putText(frame, text, (100, 100), font, 1, (255, 255, 0), 2)
        cv2.imshow('fruit recognition', frame)
        # cv2.imshow('fruit',roi)
        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break
    else:
        print('Error')

cam.release()
cv2.destroyAllWindows()
