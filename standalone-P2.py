#!/usr/bin/env python3

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import statistics
#%matplotlib inline

import math

def plot_hist_rgb(image):
    color = ('b','g','r')
    plt.figure()
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "traffic-signs-data/train.p"
validation_file="traffic-signs-data/valid.p"
testing_file =  "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Training, validation, and testing data loaded.")

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

n_train = len(X_train)

n_validation = len(X_valid)

n_test = len(X_test)

image_shape = X_train[0].shape

n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

def convert_to_YUV(array_of_RGB_images):

    yuv = np.empty_like(array_of_RGB_images)
    y = np.empty((array_of_RGB_images.shape[0], array_of_RGB_images.shape[1], array_of_RGB_images.shape[2]), dtype=np.uint8)

    for index, image in enumerate(array_of_RGB_images):
        yuv[index] = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y[index] = yuv[index][:,:,0]

    return yuv, y

# 12203 and 16516 are also nice images
images = np.array([X_train[7755], X_train[20865]])
images_yuv, images_y = convert_to_YUV(images)

pltcols = 2
pltrows = 3

plt.subplot(pltrows,pltcols,1)
plt.imshow(images[0])
plt.subplot(pltrows,pltcols,2)
plt.imshow(images[1])
plt.subplot(pltrows,pltcols,3)
plt.imshow(images_yuv[0])
plt.subplot(pltrows,pltcols,4)
plt.imshow(images_yuv[1])
plt.subplot(pltrows,pltcols,5)
plt.imshow(images_y[0])
plt.subplot(pltrows,pltcols,6)
plt.imshow(images_y[1])

plt.show()
