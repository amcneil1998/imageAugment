import cv2
import numpy as np
import imutils


class Generator():
    def __init__(self, imagePath, zoom=False, shear=False, doHorizontalFlips=False, 
            doVerticalFlips=False, augmentBrigtness=False, augmentSaturation=False, 
            addBlur=False, addNoise=False, doRotation=False):

            self.imagePath = imagePath
            self.zoom = zoom
            self.shear = shear
            self.doHorizontalFlips = doHorizontalFlips
            self.doVerticalFlips = doVerticalFlips
            self.augmentBrigtness = augmentBrigtness
            self.augmentSaturation = augmentSaturation
            self.addBlur = addBlur
            self.addNoise = addNoise
            self.doRotation = doRotation


    def augment(self):

        image = cv2.imread(self.imagePath)


        #do zoom

        #do shear

        #flip only on horizontal
        if self.doHorizontalFlips and not self.doVerticalFlips:
            image = cv2.flip(image, 0)

        #flip only on verticle
        if self.doVerticalFlips and not self.doHorizontalFlips:
            image = cv2.flip(image, 1)

        #flip both verticlly and horizontally
        if self.doVerticalFlips and self.doHorizontalFlips:
            image = cv2.flip(image, -1)


        #add blur
        if self.addBlur:
            kernelSize = int(self.addBlur * 100)
            if kernelSize % 2 == 0:
                kernelSize += 1
            image = cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
        
        #add rotation
        if self.doRotation:
            rotation = np.random.rand([0, self.doRotation])
            angle = np.arange(rotation, self.doRotation)
            rotated = imutils.rotate_bound(image, angle)
            cv2.imshow("Rotated (Correct)", rotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #change brightness
        if self.augmentBrigtness:
            cols, rows, none = image.shape
            brightness = np.sum(image[:,:,-1])/(255*cols*rows)
            image = cv2.convertScaleAbs(image, alpha=1, beta=(255*(1-brightness)))
        

    
