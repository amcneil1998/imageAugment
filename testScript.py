import Augment
testGen = Augment.Generator.createGenerator(dirPath='/Users/cap1a1n/SeniorDesign/augment/pictures/', Batch_size=2, zoom=0.2, doHorizontalFlips=False, doVerticalFlips=False, augmentBrightness=0.2, addBlur=0.2, 
addNoise=0.1, doRotation=20)