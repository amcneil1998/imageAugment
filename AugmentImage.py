import cv2
 
img = cv2.imread('/home/jakekenn97/Desktop/imageAugment/image.png', cv2.IMREAD_UNCHANGED)
print(img.shape)
height, width, depth = img.shape

zoom = 0.49

cropXStart = int(width/2 - zoom*width)
cropXEnd = int(width/2 + zoom*width)
cropYStart = int(height/2 - zoom*height)
cropYEnd = int(height/2 + zoom*height)

cropped = img[cropXStart:cropXEnd, cropYStart:cropYEnd]
newimg = cv2.resize(cropped, (img.shape[1], img.shape[0]))
print(newimg.shape)
cv2.imshow("cropped", cropped)
cv2.waitKey(0)
cv2.imshow("orig", img)
cv2.waitKey(0)