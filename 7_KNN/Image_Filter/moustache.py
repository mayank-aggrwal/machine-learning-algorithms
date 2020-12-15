
import cv2

img = cv2.imread('mustache.png', -1)
# img = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

cv2.imshow('Pic', img)
cv2.waitKey(0)
cv2.destroyAllWindows()