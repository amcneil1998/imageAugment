import cv2
import numpy as np
import imutils
import os
from random import randint

class Generator():

    #this will create a generator that yeilds images loaded from the directory specified
    #in dirPath.  Output will yeild images of size Batch_sizeximagecolsximagerowsximagedepth.
    #values specified are transfered directly into agument image
    def createGenerator(self, dirPath, Batch_size, zoom=False, shear=False, doHorizontalFlips=False, doVerticalFlips=False, augmentBrigtness=False, augmentSaturation=False, addBlur=False, addNoise=False, doRotation=False):
        nameList = os.listdir(dirPath)
        testImage = cv2.imread(nameList[0])
        cols, rows, depth = testImage.shape
        imageStorage = np.zeros((len(nameList), cols, rows, depth), dtype=np.uint8)
        imageStorage[0] = testImage
        for i in range(1, len(nameList)):
            imageStorage[i] = cv2.imread(nameList[i])
        while True:
            usedImages = np.random.randint(0, len(nameList), Batch_size)
            yield augmentImages(imageStorage[usedImages],
                                zoom=zoom,
                                shear=shear,
                                doHorizontalFlips=doHorizontalFlips,
                                doVerticalFlips=doVerticalFlips,
                                augmentBrigtness=augmentBrigtness,
                                augmentSaturation=augmentSaturation,
                                addBlur=addBlur,
                                addNoise=addNoise,
                                doRotation=doRotation)
    
    #This method will take in an array of images as well as max agumentation values
    #it will then compute a random uniform augmentation between the specified agumentation value
    #and zero. In the case of rotation, the maximum value is 360 and min is 0 in degrees. 
    #This will be done uniquely for each image.  The resulting array of agumented images is returned
    def augmentImages(self, 
                      Images, 
                      zoom=False, 
                      shear=False, 
                      doHorizontalFlips=False, 
                      doVerticalFlips=False, 
                      augmentBrigtness=False, 
                      augmentSaturation=False, 
                      addBlur=False, 
                      addNoise=False, 
                      doRotation=False):

        for i in range(0, Images.shape[0]):
            if zoom:
                zoomVal = np.random.uniform(0, zoom)
            if shear:
                shearVal = np.random.uniform(0, shear)
            if augmentBrigtness:
                brightValue = np.random.uniform(0,augmentBrigtness)
            if augmentSaturation:
                satVal = np.random.uniform(0, augmentSaturation)
            if addBlur:
                blurVal = np.random.uniform(0, addBlur)
            if addNoise:
                noiseVal = np.random.uniform(0, addNoise)
            if doRotation:
                rotateVal = np.random.uniform(0, doRotation)

            Images[i] = augment(Images[i], 
                                Zoom=zoomVal, 
                                Shear=shearVal, 
                                doHorizontalFlips=doHorizontalFlips, 
                                doVerticalFlips=doVerticalFlips, 
                                augmentBrigtness=brightValue, 
                                augmentSaturation=satVal, 
                                addBlur=blurVal, 
                                addNoise=noiseVal, 
                                doRotation=rotateVal)
        return Images

    #this method will perform all image agmentation on an image
    #input values can be either Booleans such as false or 
    #decemals representing a 0-1.0 scale change
    #ceterain values like doHorizontalFlips should always be booleans
    #whereas others such as blur and brightness dont make since to be booleans
    def augment(self, 
                image, 
                zoom=False, 
                shear=False, 
                doHorizontalFlips=False, 
                doVerticalFlips=False, 
                augmentBrigtness=False, 
                augmentSaturation=False, 
                addBlur=False, 
                addNoise=False, 
                doRotation=False):

        #do zoom
        if zoom:
            height, width, depth = image.shape
            cropXStart = int(width/2 - (1-zoom)/2*width)
            cropXEnd = int(width/2 + (1-zoom)/2*width)
            cropYStart = int(height/2 - (1-zoom)/2*height)
            cropYEnd = int(height/2 + (1-zoom)/2*height)

            cropped = image[cropYStart:cropYEnd, cropXStart:cropXEnd]
            image = cv2.resize(cropped, (image.shape[1], image.shape[0]))

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
        
        #add rotation
        if doRotation:
            image = imutils.rotate_bound(image, doRotation)

        #change brightness
        if augmentBrigtness:
            cols, rows, none = image.shape
            brightness = np.sum(image[:,:,-1])/(255*cols*rows)
            image = cv2.convertScaleAbs(image, alpha=1, beta=(255*(augmentBrigtness-brightness)))
        

    
