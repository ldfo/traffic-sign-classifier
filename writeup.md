# **Traffic Sign Recognition** 

## Write-up
---

**Build a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[top15]: ./top15.png "Top 15 sign names"
[org_image]: ./org_image.png "Original image"
[proc_image]: ./processed_image.png "Processed image"
[sign_count_plot]: ./sign_count_plot.png "Traffic sign counts plot"
[im1]: ./ts_fromtheweb/sign1.png "Traffic Sign 1"
[im2]: ./ts_fromtheweb/sign2.png "Traffic Sign 2"
[im3]: ./ts_fromtheweb/sign3.png "Traffic Sign 3"
[im4]: ./ts_fromtheweb/sign4.png "Traffic Sign 4"
[im5]: ./ts_fromtheweb/sign5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Write-up
Here is a link to my [project code](https://github.com/ldfo/traffic-sign-classifier)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32x3
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset

Here we can see the top 15 most common Sign Names with Speed limit 50km/h at the top and 5.8% of the dataset.
On the graph we can see some signs like 'Go straight or left' and 'Speed limit 20km/h' are a lot less frequent. 

![alt text][top15]
![alt text][sign_count_plot]

### Design and Test a Model Architecture

#### 1. Image processing.

First I converted the images to grayscale for reducing the dimensionality. I assume grayscale images work better because the excess information of the extra channels only adds confusion into the learning process. 

Then I used the cv2.equalizeHist function for making the histograms of the image more centered, it doesn't work well in places where we have very bright and very dark pixels but in general, it does a good job making the images more homogeneous.

Lastly I normalized the images for giving a more consistent dynamic range on the images.

If we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ (proportionally speaking) from one another. We might be over compensating a correction in one weight dimension while undercompensating in another.

Example of an image before and after processing:

![alt text][org_image]
![alt text][proc_image]


#### 3. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 black and white image   				| 
| Conv2D  		     	| 1x1 stride, VALID padding, outputs 28x28x28 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 28x28x28, padding Same	|
| Conv2D			    | 1x1 stride,  outputs 24x24x16, padding Valid	|
| RELU 					| 												|
| Max pooling			| 1x1 stride,  outputs 24x24x16, padding Same	|
| Conv2D			    | 2x2 stride,  outputs 10x10x10, padding Valid	|
| RELU 					| 												|
| Max pooling			| 1x1 stride,  outputs 10x10x10, padding Same	|
| Flatten				| size = 1000									|
| Fully connected		| size = 256,  RELU								|
| dropout				| keep probability = 0.75						|
| Fully connected		| size = 128,   RELU							|
| dropout				| keep probability = 0.75						|
| Fully connected		| size = 64,   RELU								|
| dropout				| keep probability = 0.75						|
| Fully connected (out)	| size = number of classes,   linear			|
 
#### 3. Model training

To train the model, I used a batch size of 256, 15 epochs, a learning rate of 0.00025 with an AdamOptimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.
My final model results were:
* training set accuracy of 98%
* validation set accuracy of 95% 
* test set accuracy of 93%

If an iterative approach was chosen:
* At first I chose an architecture consisting of a conv2d layer then a max_pool layer and then some fully connected layers because I thought it was a good starting point.
* I quickly achieved 85% accuracy with that architecture but it seemed that I needed more convolution and max pooling layers.
* After adding two other convolutions and max pooling layers I also added some dropout layers to help with overfitting.
* Lastly I tuned the learning rate which I found was too high at first because the accuracy was oscillating a lot.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][im1] ![alt text][im2] ![alt text][im3] 
![alt text][im5] ![alt text][im4]

The first image is a challenge because it is at an angle.

The second image is a little bit at an angle.

The third image may be a challenge because of the uneven lighting and slight slanting.

The fourth image has a number three which can be confused with an 8

Image 5 shouldn't be a problem.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| Turn right ahead		| Turn right Ahead								|
| No entry				| Dangerus curve to the right					|
| General caution		| General caution      							|
| Speed limit 30 km/h  	| Speed limit 30 km/h 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
The performance on the new images is similar to the performance on the training set, but it fails on the image that is heavily at an angle, data augmentation of the training set may help with this kind of errors.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image (ahead only) the network is correct but no vehicles follow not that far away.


| probability 	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Ahead only   									| 
| .0004					| No passing for vehicles over 3.5 metric...	|
| .0004	      			| Speed limit 60km/h			 				|

For the second image, it is also 100% sure it's a Turn right ahead

For the third image it got it wrong, the probabilities are:

| probability 	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| Yield		   									| 
| .48					| Ahead only									|
| .0024					| Bumpy road									|
| .0012	      			| Dangerous curve to the right	 				|
