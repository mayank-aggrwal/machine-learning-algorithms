
import cv2

img = cv2.imread('kick.jfif')
grayImg = cv2.imread('kick.jfif', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Kick Image', img)
cv2.imshow('Gray Kick Image', grayImg)

cv2.waitKey(10000)
cv2.destroyAllWindows()