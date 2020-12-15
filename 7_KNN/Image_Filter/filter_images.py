
import cv2


def overlay_transparent(background_img, img_to_overlay_t, glass, x, y, z, v, overlay_size=None, os=None):
#     """
# 	@brief      Overlays a transparant PNG onto another image using CV2

# 	@param      background_img    The background image
# 	@param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
# 	@param      x                 x location to place the top-left corner of our overlay
# 	@param      y                 y location to place the top-left corner of our overlay
# 	@param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
	
# 	@return     Background image with overlay on top
# 	"""

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
        glass = cv2.resize(glass.copy(), os)

# 	Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    bg,gg,rg,ag = cv2.split(glass)
    overlay_color = cv2.merge((b,g,r))
    overlay_colorg = cv2.merge((bg,gg,rg))

# 	Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a,5)
    maskg = cv2.medianBlur(ag,5)

    h, w, _ = overlay_color.shape
    hg, wg, _g = overlay_colorg.shape
    roi = bg_img[y:y+h, x:x+w]
    roig = bg_img[v:v+hg, z:z+wg]

# 	Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
    img1_bgg = cv2.bitwise_and(roig.copy(),roig.copy(),mask = cv2.bitwise_not(maskg))

# 	Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)
    img2_fgg = cv2.bitwise_and(overlay_colorg,overlay_colorg,mask = maskg)

    # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    bg_img[v:v+hg, z:z+wg] = cv2.add(img1_bgg, img2_fgg)

    return bg_img

# cv2.imshow('image',overlay_transparent(img, overlay_t, 0, 0, (200,200)))



# cap = cv2.VideoCapture(0)
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
m = cv2.imread('mustache.png', -1)
g = cv2.imread('glasses.png', -1)
i = cv2.imread('Jamie_Before.jpg')


grayFrame = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

noses = nose_cascade.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 9, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
eyes = eye_cascade.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 3, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
print('noses:', len(noses))
print('eyes:', len(eyes))


# cv2.imshow('Pic',overlay_transparent(i, m, noses[0][0] - 4, noses[0][1] + noses[0][3] - 52, (140,80)))
# cv2.imshow('Pic',overlay_transparent(i, g, eyes[0][0] - 17, eyes[0][1] - 15, (400,200)))
cv2.imshow('Pic',overlay_transparent(i, g, m, eyes[0][0] - 13, eyes[0][1] - 20, noses[0][0] - 4, noses[0][1] + noses[0][3] - 52, (400,215), (140,80)))

cv2.waitKey(0)
cv2.destroyAllWindows()