import cv2
import numpy as np
import os
from random import randint
import time
import math

class Generator():

    #this will create a generator that yeilds images loaded from the directory specified
    #in dirPath.  Output will yeild images of size Batch_sizeximagecolsximagerowsximagedepth.
    #values specified are transfered directly into agument image
    @staticmethod
    def createGenerator(dirPath, Batch_size, zoom=False, doHorizontalFlips=False, doVerticalFlips=False, augmentBrightness=False, addBlur=False, addNoise=False, doRotation=False):
        nameList = os.listdir(dirPath)
        for i in range(0, len(nameList)):
            nameList[i] = dirPath + nameList[i]
        testImage = cv2.imread(nameList[0])
        cols, rows, depth = testImage.shape
        imageStorage = np.zeros((len(nameList), cols, rows, depth), dtype=np.uint8)
        imageStorage[0] = testImage
        for i in range(1, len(nameList)):
            imageStorage[i] = cv2.imread(nameList[i])
        while True:
            usedImages = np.random.randint(0, len(nameList), Batch_size)
            yield Generator.augmentImages(Images=imageStorage[usedImages],
                                zoom=zoom,
                                doHorizontalFlips=doHorizontalFlips,
                                doVerticalFlips=doVerticalFlips,
                                augmentBrightness=augmentBrightness,
                                addBlur=addBlur,
                                addNoise=addNoise,
                                doRotation=doRotation)
    @staticmethod
    #this method will allow you to test the generator parameters and view some images that have been augmented
    def testGenerator(dirPath, numImages, zoom=False, doHorizontalFlips=False, doVerticalFlips=False, augmentBrightness=False, addBlur=False, addNoise=False, doRotation=False):
        nameList = os.listdir(dirPath)
        for i in range(0, len(nameList)):
            nameList[i] = dirPath + nameList[i]
        testImage = cv2.imread(nameList[0])
        cols, rows, depth = testImage.shape
        imageStorage = np.zeros((len(nameList), cols, rows, depth), dtype=np.uint8)
        imageStorage[0] = testImage
        for i in range(1, len(nameList)):
            imageStorage[i] = cv2.imread(nameList[i])
        usedImages = np.random.randint(0, len(nameList), numImages)
        return Generator.augmentImages(Images=imageStorage[usedImages],
                            zoom=zoom,
                            doHorizontalFlips=doHorizontalFlips,
                            doVerticalFlips=doVerticalFlips,
                            augmentBrightness=augmentBrightness,
                            addBlur=addBlur,
                            addNoise=addNoise,
                            doRotation=doRotation)
    
    #This method will take in an array of images as well as max agumentation values
    #it will then compute a random uniform augmentation between the specified agumentation value
    #and zero. In the case of rotation, the maximum value is 360 and min is 0 in degrees. 
    #This will be done uniquely for each image.  The resulting array of agumented images is returned
    @staticmethod
    def augmentImages(Images, 
                      zoom=False,
                      doHorizontalFlips=False, 
                      doVerticalFlips=False, 
                      augmentBrightness=False,
                      addBlur=False, 
                      addNoise=False, 
                      doRotation=False):

        for i in range(0, Images.shape[0]):
            #randomize our agmentation values
            if doHorizontalFlips:
                flipVal = np.random.uniform(0, 1)
                if flipVal < 0.5:
                    hFlipVal = True
                else:
                    hFlipVal = False
            else:
                hFlipVal = False
            if doVerticalFlips:
                flipVal = np.random.uniform(0, 1)
                if flipVal < 0.5:
                    vFlipVal = True
                else:
                    vFlipVal = False
            else:
                vFlipVal = False
            if zoom:
                zoomVal = np.random.uniform(0, zoom)
            else:
                zoomVal = False
            if augmentBrightness:
                brightValue = np.random.uniform(-augmentBrightness,augmentBrightness)
            else:
                brightValue = False
            if addBlur:
                blurVal = np.random.uniform(0, addBlur)
            else:
                blurVal = False
            if addNoise:
                noiseVal = np.random.uniform(0, addNoise)
            else:
                noiseVal = False
            if doRotation:
                rotateVal = np.random.uniform(-doRotation, doRotation)
                #normalize to 360 degree rotations
                if rotateVal < 0:
                    rotateVal = 360 - rotateVal
            else:
                rotateVal = False
            #augment the images
            Images[i] = Generator.augment(image=Images[i], 
                                zoom=zoomVal,
                                doHorizontalFlips=hFlipVal, 
                                doVerticalFlips=vFlipVal, 
                                augmentBrightness=brightValue,
                                addBlur=blurVal, 
                                addNoise=noiseVal, 
                                doRotation=rotateVal)
        return Images.astype(int)/256

    #this method will perform all image agmentation on an image
    #input values can be either Booleans such as false or 
    #decemals representing a 0-1.0 scale change
    #ceterain values like doHorizontalFlips should always be booleans
    #whereas others such as blur and brightness dont make since to be booleans
    @staticmethod
    def augment(image, 
                zoom=False,
                doHorizontalFlips=False, 
                doVerticalFlips=False, 
                augmentBrightness=False, 
                addBlur=False, 
                addNoise=False, 
                doRotation=False):
        #do zoom
        if zoom:
            height, width = image.shape[0:2]
            cropXStart = int(width/2 - (1-zoom)/2*width)
            cropXEnd = int(width/2 + (1-zoom)/2*width)
            cropYStart = int(height/2 - (1-zoom)/2*height)
            cropYEnd = int(height/2 + (1-zoom)/2*height)

            cropped = image[cropYStart:cropYEnd, cropXStart:cropXEnd]
            image = cv2.resize(cropped, (image.shape[1], image.shape[0]))

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
            image_height, image_width = image.shape[0:2]
            image_rotated = rotate_image(image, doRotation)
            image = crop_around_center(
                image_rotated,
                *largest_rotated_rect(
                    image_width,
                    image_height,
                    math.radians(doRotation)
                    )
                )
            image = cv2.resize(image, (image_width, image_height))


        #change brightness
        if augmentBrightness:
            #honestly the -40 and /4 are kinda just randomly chosen to make it a bit more consistent 
            if augmentBrightness < 0:
                image = np.clip(image.astype(int) + 255*(augmentBrightness) - 40, 0, 255).astype(np.uint8)
            else:
                cols, rows = image.shape[0:2]
                brightness = np.sum(image[:,:,-1])/(255*cols*rows)
                image = np.clip(image.astype(int) + 255*(augmentBrightness + brightness)/4, 0, 255).astype(np.uint8)
        
        #add noise
        if addNoise:
            noiseArray = np.random.random_sample(image.shape)
            image[np.where(noiseArray < addNoise)] = 0
        return image
            
def show(image):
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]