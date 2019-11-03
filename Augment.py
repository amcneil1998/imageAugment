import cv2
import numpy as np
import imutils
import os
from random import randint

#this method will perform all image agmentation on an image
#input values can be either Booleans such as false or 
#decemals representing a 0-1.0 scale change
#ceterain values like doHorizontalFlips should always be booleans
#whereas others such as blur and brightness dont make since to be booleans
def augment(imagePath, Zoom=False, Shear=False, doHorizontalFlips=False, doVerticalFlips=False, augmentBrigtness=False, augmentSaturation=False, addBlur=False, addNoise=False):
    image = cv2.imread(imagePath)

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

        for filename in os.listdir(self.imagePath):
            image = cv2.imread(filename)


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
                rotation = randint(0,self.doRotation)
                image = imutils.rotate_bound(image, rotation)

            #change brightness
            if self.augmentBrigtness:
                cols, rows, none = image.shape
                brightness = np.sum(image[:,:,-1])/(255*cols*rows)
                image = cv2.convertScaleAbs(image, alpha=1, beta=(255*(1-brightness)))

            
        

    
