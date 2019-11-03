import cv2
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


    #add blur
    if addBlur:
        kernelSize = int(addBlur * 100)
        if kernelSize % 2 == 0:
            kernelSize += 1
        image = cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
    
