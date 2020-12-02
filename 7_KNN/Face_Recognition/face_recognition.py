
# Recognise faces using classification algorithm - KNN, SVM, Logistic etc.
# --------------------------------------------------------------------------

# 1. Load the training data (numpy arrays of all the persons)
        # x - values are stored in numpy arrays
        # y - values we need to assign for each person
# 2. Read a video stream using OpenCV
# 3. Extract faces out of it
# 4. Use KNN to find the prediction of face (integer)
# 5. Map the predicted id to the name of the user
# 6. Display the predictions on the screen - bounding box and name



import cv2
import numpy as np
import os

########## KNN algorithm ####################

def dist(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def knn(train, query_pt, k=5):
    vals = []
    
    
    Y = train[:, -1]
    X = train[:, :-1]
    m = X.shape[0]
    
    for i in range(m):
        d = dist(X[i], query_pt)
        vals.append([d, Y[i]])
        
    vals = sorted(vals)
    
    vals = np.array(vals)
    vals = vals[:k, :]
    
    counts = np.unique(vals[:, 1], return_counts=True)
    idx = counts[1].argmax()
    pred = counts[0][idx]
    return pred

#############################################


# Init camera
cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

face_data = []
labels = []
dataset_path = './data/'

class_id = 0
names = {}          # Mapping for id and name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        
        target = class_id * np.ones((data_item.shape[0],)).reshape((-1, 1))
        names[class_id] = fx[:-4]
        class_id += 1
        labels.append(target)
    
face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0)

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis = 1)
print(trainset.shape)


while True:
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.3, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    for face in faces:
        x, y, w, h = face

        # Get face Region of Interest
        offset = 10
        face_section = frame[y - offset: y + h + offset, x - offset: x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        
        # Predicted label
        pred = knn(trainset, face_section.flatten())
        
        # Display the name and the rectangle on the screen
        pred_name = names[int(pred)]
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        
    cv2.imshow('Video Frame', frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






