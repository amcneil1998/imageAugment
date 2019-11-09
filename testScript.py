import cv2
import Augment
testImages = Augment.Generator.testGenerator(dirPath='/home/cap1a1n/SeniorDesign/ImageAugment/picturescopy/', numImages=256, zoom=0.1, doHorizontalFlips=True, doVerticalFlips=True, augmentBrightness=0.1, addBlur=0.1, 
addNoise=0.05, doRotation=20)
'''
for i in range(0, len(testImages)):
    cv2.imshow("img", testImages[i]); cv2.waitKey(0); cv2.destroyAllWindows()
'''