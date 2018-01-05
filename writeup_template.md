# **Traffic Sign Recognition** 

## Writeup for Chris Svec

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[training_hist]: ./report-images/orig-training-data-histogram.png "original training data"
[aug_training_hist]: ./report-images/augmented-training-data-histogram.png "augmented training data"
[sign_20_kph_rgb]: ./report-images/sign-20-kph-rgb.png "original RGB image"
[sign_20_kph_gray]: ./report-images/sign-20-kph-gray.png "grayscale image"
[sign_20_kph_gray_shifted_1]: ./report-images/sign-20-kph-gray-shifted-1.png "shifted example 1"
[sign_20_kph_gray_shifted_2]: ./report-images/sign-20-kph-gray-shifted-2.png "shifted example 2"
[stop]: ./images-from-the-web/stop.png "stop sign"
[caution]: ./images-from-the-web/caution-1.png "caution sign"
[priority_road]: ./images-from-the-web/Arterial-priority.png "priority road sign"
[no_entry]: ./images-from-the-web/Do-Not-Enter.png "no entry sign"
[right_of_way]: ./images-from-the-web/right-of-way.png "right of way sign"
[speed_100]: ./images-from-the-web/speed-100.png "100 kph sign"
[caution_tricky]: ./images-from-the-web/caution.png "tricky caution sign"
[crossing]: ./images-from-the-web/elderly-crossing.png "tricky crossing sign"
[speed_130]: ./images-from-the-web/speed-130.png "tricky 130 kph sign"
[work_speed_30]: ./images-from-the-web/work-and-speed-30.png "tricky work and speed sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/svec/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of the training set is: 34799 images
* The size of the validation set is: 4410 images
* The size of the test set is: 12630 images
* The shape of a traffic sign image is: 32x32x3 pixels
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

I plotted a histogram of the training, validation, and test sets to see how representative they were
of the 43 different classes of images:

![alt text][training_hist]

The validation and test data looked similar. You can see that the 43 images classes are not
represented uniformly: some have fewer than 200 images (0.5% of the data set), whereas some have
around 2000 (5.7%). If the image classes were represented uniformly each image class would make up
about 2.3% of the data set.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first I did no preprocessing and used the LeNet network architecture, modified only for the different
input and output parameters. I did this to make sure I understood what the data looked like and that
the basic TensorFlow setup worked. This gave me a training accuracy for 98% - but a validation
accuracy of only 86%, so clearly there was work to be done.

I started by reading Sermanet and LeCun's baseline paper, ["Traffic Sign Recognition with Multi-Scale Convolutional Networks"]([http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper, which was surprisingly understandable thanks to this SDC class!

I attempted to convert the RGB images to YUV as the paper had done, but couldn't get it to work; I
ran into some matrix/array sizing issues that were probably due to misunderstanding how the
conversion worked.

I settled for converting to grayscale instead. I also did a poor-man's normalization on the grayscale image using
(128-pixel)/128 to center it at 0 as suggested by the project.

I decided to generate additional data since the quality of the neural network's predictions are only
as good as the training data. Too many of the sign classes had too few examples in the training
data, so I generated additional images for each image class that had fewer than 1000 images by
taking the grayscale images and shifting them by 1-4 pixels vertically and horizontally.

Here is an example of an original RGB sign:

![alt text][sign_20_kph_rgb]

The grayscale + normalized image:

![alt text][sign_20_kph_gray]

And 2 randomly shifted variants that were generated as extra data:

![alt text][sign_20_kph_gray_shifted_1]
![alt text][sign_20_kph_gray_shifted_2]

I chose to gray-out the parts of the image that were "shifted out" of the image by setting the pixel
value equal to the mean value in the image.

After adding the new images, you can see that the class histograms show a better representation of
all classes:

![alt text][training_hist]
![alt text][aug_training_hist]

Additionally I tried blurring the images using a Gaussian blur, but saw no significant accuracy
changes, so I did not use blurring.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the original LeNet architecture, with two additional dropout layers to prevent
overfitting.

It looks like:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten               | 5x5x16 -> 400 |
| Fully connected		| input 400, output 120 |
| RELU					|												|
| Dropout 1             | 50% dropout |
| Fully connected		| input 120, output 84 |
| RELU					|												|
| Dropout 2             | 50% dropout |
| Fully connected output | input 84, output 43 |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the default values from the original project and the LeNet example,
except I changed the number of epochs to 20.

I tried varying the learning rate, sigma, and batch size but didn't see much difference from run to
run.

Switching to 20 epochs allowed the validation rate to get up around 95% for the last 1 or 2 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97.1%
* validation set accuracy of 94.1%
* test set accuracy of 92.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * I started with the LeNet architecture from the previous lesson, with the required input/output
    size tweaks. I started here because the intro video to the project led us from LeNet to a
    starting point for the traffic sign problem.
* What were some problems with the initial architecture?
  * The original LeNet architecture had too low of a validation accuracy (86%), and a high training
    accuracy (98%), indicating overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc.), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * My first adjustment was to add a dropout layer to help prevent overfitting. I ended up adding
    two layers with 50% dropout each.
  * I tried modifying my LeNet-based architecture to look more like the LeCun paper's network by
    removing convolutional layers and using cross-layer connections, but didn't see gains.
    * I suspect I wasn't changing the architecture correctly, but I ran out of time for the project
      and was seeing good enough results with my LeNet-based architecture.
* Which parameters were tuned? How were they adjusted and why?
  * I tried several variants of the dropout layers and dropout percentages to get a high enough
    validation accuracy and reduce overfitting.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * A convolutional layer probably works well for this problem because the signs are differentiated
    by how parts of the images look. The convolutional layers learn the sub-image characterisics and
    map them to the correct image class most of the time.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose 10 German traffic signs: 6 that should be easily identifiable, and 5 that should not. All
were shrunk to 32x32 for processing.

Here are the six hopefully easy to identify signs:

1. ![alt_text][stop]
2. ![alt_text][caution]
3. ![alt_text][priority_road]
4. ![alt_text][no_entry]
5. ![alt_text][right_of_way]
6. ![alt_text][speed_100]

The four signs that should be difficult or impossible to identify are:

7. ![alt_text][caution_tricky]
8. ![alt_text][work_speed_30]
9. ![alt_text][speed_130]
10. ![alt_text][crossing]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1. stop           	| stop |
| 2. caution            | caution |
| 3. priority road      | priority road |
| 4. no entry           | no entry |
| 5. right of way       | right of way |
| 6. 100 kph            | 20 kph |
| 7. caution with another sign under it | bumpy road |
| 8. road work and 30 kph | go straight or right |
| 9. 130 kph            | 30 kph |
| 10. elderly crossing  | children crossing |


The model was able to correctly guess 6 of the 10 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Jupyter notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Jupyter notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


