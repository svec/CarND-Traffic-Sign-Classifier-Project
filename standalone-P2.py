#!/usr/bin/env python3

#Training, Validation Accuracy = 0.506 0.456
#Training, Validation Accuracy = 0.710 0.625
#Training, Validation Accuracy = 0.807 0.727
#Training, Validation Accuracy = 0.869 0.778
#Training, Validation Accuracy = 0.909 0.805
#Training, Validation Accuracy = 0.920 0.818
#Training, Validation Accuracy = 0.938 0.845
#Training, Validation Accuracy = 0.948 0.859
training_accuracy = [0.506, 0.710, 0.807, 0.869, 0.909, 0.920, 0.938, 0.948] 
validation_accuracy = [0.456, 0.625, 0.727, 0.778, 0.805, 0.818, 0.845, 0.859]

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import statistics
import random
#%matplotlib inline

import math

def plot_with_labels(data, label):
    xes = list(range(1, len(data)+1))
    plt.plot(xes, data, label=label, marker='o')
    plt.ylim(0,1)
    #plt.xlim(1,len(data)+1)
    plt.legend()
    for index,data_value in zip(xes, data):
        plt.annotate(str(round(data_value,2)), xy=(index, data_value), xytext=(10,10), textcoords='offset points')

def test_acc():
    plot_with_labels(training_accuracy, "training")
    plot_with_labels(validation_accuracy, "validation")
    plt.grid(True)
    plt.xlabel("epoch")
    plt.show()

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

def convert_to_gray(array_of_RGB_images):

    gray = np.empty((array_of_RGB_images.shape[0], array_of_RGB_images.shape[1], array_of_RGB_images.shape[2]))

    for index, image in enumerate(array_of_RGB_images):
        gray[index] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = np.expand_dims(gray, axis=3)

    return gray


def convert_to_YUV(array_of_RGB_images):

    yuv = np.empty_like(array_of_RGB_images)
    y = np.empty((array_of_RGB_images.shape[0], array_of_RGB_images.shape[1], array_of_RGB_images.shape[2]), dtype=np.uint8)

    for index, image in enumerate(array_of_RGB_images):
        yuv[index] = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y[index] = yuv[index][:,:,0]

    return yuv, y

def test_yuv():
    # 12203 and 16516 are also nice images
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

def test_gray():
    images_gray = convert_to_gray(images)

    print("images_gray shape:", images_gray.shape)
    pltcols = 2
    pltrows = 2
    
    plt.subplot(pltrows,pltcols,1)
    plt.imshow(images[0])
    plt.subplot(pltrows,pltcols,2)
    plt.imshow(images[1])

    plt.subplot(pltrows,pltcols,3)
    plt.imshow(images_gray[0].squeeze(), cmap='gray')
    plt.subplot(pltrows,pltcols,4)
    plt.imshow(images_gray[1].squeeze(), cmap='gray')
    plt.show()

def test_shifting(images):
    num_images_to_create = 5

    shifted = np.empty((10, images.shape[1], images.shape[2], images.shape[3]))

    print("image, shifted shape:", images[0].shape, shifted.shape)

    possible_offsets = (-4, -3, -2, -1, 1, 2, 3, 4)

    for ii in range(num_images_to_create):
        y_offset = random.choice(possible_offsets)
        x_offset = random.choice(possible_offsets)
        shifted[ii] = np.roll(images[0], (y_offset, x_offset), axis=(0,1))
        # np.roll() shifts the pixels around the edge of the image, so we need to
        # set the pixels that rolled around the edge to black.
        if y_offset > 0:
            clear_y_start = 0
            clear_y_stop = y_offset
        else:
            clear_y_start = y_offset
            clear_y_stop = shifted[ii].shape[0]
        if x_offset > 0:
            clear_x_start = 0
            clear_x_stop = x_offset
        else:
            clear_x_start = x_offset
            clear_x_stop = shifted[ii].shape[1]

        shifted[ii][clear_y_start:clear_y_stop:1,:,:] = 255
        shifted[ii][:,clear_x_start:clear_x_stop:1,:] = 255
        print("y, x offset:", y_offset, x_offset)

    pltcols = 2
    pltrows = num_images_to_create+1
    
    plt.subplot(pltrows,pltcols,1)
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.subplot(pltrows,pltcols,2)
    plt.imshow(images[1].squeeze(), cmap='gray')

    plt.subplot(pltrows,pltcols,3)
    plt.imshow(shifted[0].squeeze(), cmap='gray')
    plt.subplot(pltrows,pltcols,5)
    plt.imshow(shifted[1].squeeze(), cmap='gray')
    plt.subplot(pltrows,pltcols,7)
    plt.imshow(shifted[2].squeeze(), cmap='gray')
    plt.subplot(pltrows,pltcols,9)
    plt.imshow(shifted[3].squeeze(), cmap='gray')
    plt.subplot(pltrows,pltcols,11)
    plt.imshow(shifted[4].squeeze(), cmap='gray')
    plt.show()


images = np.array([X_train[7755], X_train[20865]])
images_gray = convert_to_gray(images)
test_shifting(images_gray)
