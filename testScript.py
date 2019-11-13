import Augment
testGen = Augment.Generator.createGenerator(dirPath='/home/jakekenn97/Desktop/imageAugment/final/', Batch_size=2, zoom=0.1, doHorizontalFlips=True, doVerticalFlips=False, augmentBrightness=1, addBlur=0.2, 
addNoise=0.1, doRotation=20)