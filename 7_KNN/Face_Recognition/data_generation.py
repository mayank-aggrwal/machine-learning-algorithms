
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './data/'

file_name = input('Enter person name: ')

while True:
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.3, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    faces = sorted(faces, key = lambda f : f[2] * f[3], reverse = True)
    
    # Largest face
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

        # Extract (Crop out the required face) : Region of Interest
        offset = 10
        face_section = frame[y - offset: y + h + offset, x - offset: x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        
        skip += 1
        if skip % 10 == 0:
            print('Taken')
            face_data.append(face_section)

        
        cv2.imshow('Face section', face_section)
        
    cv2.imshow('Collect data', frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

        
# Convert face_data list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save this data into filesystem
np.save(dataset_path + file_name + '.npy', face_data)
cap.release()
cv2.destroyAllWindows()


























