import cv2

#this method will perform all image agmentation on an image
#input values can be either Booleans such as false or 
#decemals representing a 0-1.0 scale change
#ceterain values like doHorizontalFlips should always be booleans
#whereas others such as blur and brightness dont make since to be booleans
def augment(imagePath, Zoom=False, Shear=False, doHorizontalFlips=False, doVerticalFlips=False, augmentBrigtness=False, augmentSaturation=False, addBlur=False, addNoise=False):
    image = cv2.imread(imagePath)


    #do zoom

    #do shear

    #flip only on horizontal
    if doHorizontalFlips and not doVerticalFlips:
        image = cv2.flip(image, 0)

    #flip only on verticle
    if doVerticalFlips and not doHorizontalFlips:
        image = cv2.flip(image, 1)

    #flip both verticlly and horizontally
    if doVerticalFlips and doHorizontalFlips:
        image = cv2.flip(image, -1)

    #change brightness
    if augmentBrigtness:
        cols, rows, none = image.shape
        brightness = np.sum(image[:,:,-1])/(255*cols*rows)
        image = cv2.convertScaleAbs(image, alpha=1, beta=(255*(1-brightness)))
    
    #add blur
    if addBlur:
        kernelSize = int(addBlur * 100)
        if kernelSize % 2 == 0:
            kernelSize += 1
        image = cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
    