#!/usr/bin/env python3
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed

from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
import random

# Load pickled data
import pickle
import numpy as np

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
    
X_train_rgb, y_train = train['features'], train['labels']
X_valid_rgb, y_valid = valid['features'], valid['labels']
X_test_rgb, y_test = test['features'], test['labels']

print("Training, validation, and testing data loaded.")

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train_rgb)

# TODO: Number of validation examples
n_validation = len(X_valid_rgb)

# TODO: Number of testing examples.
n_test = len(X_test_rgb)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train_rgb[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import sys
from collections import Counter
counts = Counter(y_train)
#for sign_class_number in range(n_classes):
    #print("sign class: {:2d}  count: {:5d}".format(sign_class_number, counts[sign_class_number]))

#X_train_rgb, y_train = shuffle(X_train_rgb, y_train)

random.seed(42)

g_subplot_cols = 0
g_subplot_rows = 0
g_subplot_current = 0

def subplot_setup(cols, rows):
    global g_subplot_cols
    global g_subplot_rows
    global g_subplot_current
    g_subplot_cols = cols
    g_subplot_rows = rows
    g_subplot_current = 1
    #plt.figure()


def subplot_next(image):
    global g_subplot_cols
    global g_subplot_rows
    global g_subplot_current

    if g_subplot_current == 0:
        print("ERROR: subplot_next called before subplot_setup")
        sys.exit(1)

    if g_subplot_current > (g_subplot_cols * g_subplot_rows):
        print("ERROR: too many subplots for rows, cols:", g_subplot_rows, g_subplot_cols)
        sys.exit(1)

    plt.subplot(g_subplot_rows, g_subplot_cols, g_subplot_current)
    plt.imshow(image.squeeze(), cmap='gray')

    g_subplot_current = g_subplot_current + 1

def modify_image(image):

    possible_offsets = (-4, -3, -2, -1, 1, 2, 3, 4)

    y_offset = random.choice(possible_offsets)
    x_offset = random.choice(possible_offsets)
    modified = np.roll(image, (y_offset, x_offset), axis=(0,1))
    #print("image, shifted shape:", image.shape, modified.shape)

    # np.roll() shifts/wraps pixels around the edge of the image, which results
    # in impossible input data. Cheap (and hacky) solution: gray out the pixels
    # that get wrapped.
    if y_offset > 0:
        clear_y_start = 0
        clear_y_stop = y_offset
    else:
        clear_y_start = y_offset
        clear_y_stop = modified.shape[0]
    if x_offset > 0:
        clear_x_start = 0
        clear_x_stop = x_offset
    else:
        clear_x_start = x_offset
        clear_x_stop = modified.shape[1]

    #print("y, x offset:", y_offset, x_offset)
    #print("clear y start, stop:", clear_y_start, clear_y_stop, "clear x start, stop", clear_x_start, clear_x_stop)

    original_mean_pixel_value = np.mean(modified)

    modified[clear_y_start:clear_y_stop:1,:,:] = original_mean_pixel_value
    modified[:,clear_x_start:clear_x_stop:1,:] = original_mean_pixel_value

    return modified

def add_extra_images():
    global X_train_actual
    global y_train
    global n_classes
    global counts

    min_images = 500

    X_train_actual_start_concat_index = n_train

    # Gray problem with class 19, 20

    for sign_class_number in range(n_classes):
        sign_class_counts = counts[sign_class_number]
        print("sign class: {:2d}  count: {:5d}".format(sign_class_number, sign_class_counts))
        if sign_class_counts < min_images:

            image_count_to_create = min_images - sign_class_counts
            new_images = np.empty((image_count_to_create, X_train_actual.shape[1], X_train_actual.shape[2], X_train_actual.shape[3]))

            print("creating {} extra images for class {}".format(image_count_to_create, sign_class_number))
            
            # Walk through the original signs, create a modified version of each one until
            # image_count_to_create new modified images are created.
            indices_for_this_class = np.where(sign_class_number == y_train)[0]
            original_sign_count = len(indices_for_this_class)
            original_sign_index = 0

            for extra in range(image_count_to_create):
                #print("using original index:", original_sign_index, "X_train_actual index:", indices_for_this_class[original_sign_index])
                new_image = modify_image(X_train_actual[indices_for_this_class[original_sign_index]])
                original_sign_index = (original_sign_index + 1) % original_sign_count
                new_images[extra] = new_image

            X_train_actual = np.concatenate((X_train_actual, new_images))

            modified_orig_count = int(extra / original_sign_count) + 1
            #print("class:", sign_class_number)
            #print("modified_orig_count = (extra / original_sign_count) + 1= {} = {} / {}".format(modified_orig_count, extra, original_sign_count))
            subplot_setup(cols=2,rows=1+modified_orig_count)
            subplot_next(X_train_actual[indices_for_this_class[0]])
            subplot_next(X_train_actual[indices_for_this_class[1]])
            print("original means: {:.3f} {:.3f}".format(np.mean(X_train_actual[indices_for_this_class[0]]), np.mean(X_train_actual[indices_for_this_class[1]])))
            print("original max:   {:.3f} {:.3f}".format(np.max(X_train_actual[indices_for_this_class[0]]), np.mean(X_train_actual[indices_for_this_class[1]])))
            print("original min:   {:.3f} {:.3f}".format(np.min(X_train_actual[indices_for_this_class[0]]), np.mean(X_train_actual[indices_for_this_class[1]])))
            for ii in range(modified_orig_count):
                subplot_next(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii])
                subplot_next(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii+1])
                print("new means:      {:.3f} {:.3f}".format(np.mean(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii]), np.mean(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii+1])))
                print("new max:        {:.3f} {:.3f}".format(np.max(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii]), np.mean(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii+1])))
                print("new min:        {:.3f} {:.3f}".format(np.min(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii]), np.mean(X_train_actual[X_train_actual_start_concat_index+(sign_class_counts*ii)+ii+1])))

            X_train_actual_start_concat_index = X_train_actual_start_concat_index + image_count_to_create
            plt.show()
            
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

X_train_gray = convert_to_gray(X_train_rgb)
X_valid_gray = convert_to_gray(X_valid_rgb)
X_test_gray = convert_to_gray(X_test_rgb)

# Poor-man's normalization.
X_train_gray = (X_train_gray-128)/128
X_valid_gray = (X_valid_gray-128)/128
X_test_gray  = (X_test_gray-128)/128

# Convert to YUV
#X_train_yuv, X_train_y = convert_to_YUV(X_train_rgb)
#X_valid_yuv, X_valid_y = convert_to_YUV(X_valid_rgb)
#X_test_yuv, X_test_y = convert_to_YUV(X_test_rgb)

#X_train_yuv = (X_train_yuv - 128) / 128
#X_valid_yuv = (X_valid_yuv - 128) / 128
#X_test_yuv = (X_test_yuv - 128) / 128

#X_train_actual = X_train_rgb
#X_valid_actual = X_valid_rgb
#X_test_actual =  X_test_rgb
X_train_actual = X_train_gray
X_valid_actual = X_valid_gray
X_test_actual =  X_test_gray

print("shape of X_train_actual, y_train:", X_train_actual.shape, y_train.shape)
if len(X_train_actual.shape) > 3:
    input_color_depth = X_train_actual.shape[3]
else:
    input_color_depth = 1
print("color depth:", input_color_depth)

add_extra_images()

print("final shape of X_train_actual, y_train:", X_train_actual.shape, y_train.shape)

print('done')
