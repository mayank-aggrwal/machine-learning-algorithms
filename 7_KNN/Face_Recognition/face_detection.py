
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Video Frame', frame)
    
    q = cv2.waitKey(1)
#     print(q)
    key_pressed = q & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()