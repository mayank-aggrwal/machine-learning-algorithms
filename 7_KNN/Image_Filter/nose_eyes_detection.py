
import cv2

cap = cv2.VideoCapture(0)
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')

while True:
    ret, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue
        
    noses = nose_cascade.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 9, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    eyes = eye_cascade.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 3, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in noses:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Video Frame', frame)
    
    q = cv2.waitKey(1)
#     print(q)
    key_pressed = q & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()