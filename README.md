# Introduction
This repository holds a script that is useful to do image augmentation, with the idea to be used as a generator for a Neural Netowrk.  The main script is called Augment.py and can be imported with the simple import: 

```
from imageAugment import Augment
```

This package requires the following libraries to be installed in your environment:
 
* Numpy

* OpenCV


The Augment.py class one class with four static methods highlighted in the following sections.

# createGenerator
This method can be called with the following python code:

```
Augment.createGenerator(arguments)
```

This method takes the following arguments:

* dirPath- Location of your training images.

* Batch_size- Batch size you want to train with.

* zoom- Default False, or 0-1 zoom scale.

* doHorizontalFlips- Default False, controls if the generator is allowed to horizontally flip images.

* doVerticalFlips- Default False, controls if the generator is allowed to vertically flip images.

* augmentBrightness- Default to False, takes a 0-1 representing percent.

* addBlur- Default to False, takes a 0-1 value representing percent.

* addNoise- Default to False, takes a 0-1 value representing the probability of a pixel getting salt and pepper noise

* doRotation- Default to False, takes a 0-180 value representing degree value for the maximum allowed rotation of an image.

This method returns a generator.

# testGenerator
This method can be called with the following python code:

```
Augment.createGenerator(arguments)
```

It takes the same arguments as the createGenerator method, except Batch_size is replaced with number of images.

This method returns specified number of images that have been augmented.

# augmentImages
This method can be called with the following python code:

```
Augment.createGenerator(arguments)
```

It has the following arguments: 

* Images- Array of images to be augmented

* zoom, doHorizontalFlips, doVerticalFlips, augmentBrightness, addBlur, addNoise, and doRotation, which perform the same jobs they do in createGenerator.  

This method is responsible for the randomness in the augmentation. It returns an array of images augmented randomly with the specified parameters.

# augment
This method is responsible for doing the actual image augmentation.  It has the same inputs as augmentImages, except it takes a single image.  In addition the values passed in are the actual augmentation values used.

# Additional Repository Files
* Pictures(directory)- Some sample images to augment, has regular pictures and greyscale pictures called mask.

* ```__init__.py```- Python import file.
