# **Traffic Sign Recognition** 

## Writeup
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
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup
Here is a link to my [project code](https://github.com/ldfo/traffic-sign-classifier)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset

Here we can see the top 15 most common Sign Names with Speed limit 50km/h at the top and 5.8% of the dataset.
On the graph we can see some signs like 'Go straight or left' and 'Speed limit 20km/h' are a lot less frequent. 

![alt text][top15]
![alt text][sign_count_plot]

### Design and Test a Model Architecture

#### 1. Image processing.

First I converted the images to grayscale for reducing the dimensionality. We want to train only on one color channel because it's enough.

Then I used the cv2.equalizeHist function for making the histograms of the image more centered, it doesn't work well in places where we have very bright and very dark pixels but in general it does a good job making the images more homogeneous.

Lastly I normalized the images for giving a more consistent dynamic range on the images.

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
| Max pooling	      	| 1x1 stride,  outputs 14x14x16, padding Same	|
| Conv2D			    | 1x1 stride,  outputs 10x10x8, padding Valid	|
| RELU 					| 												|
| Max pooling			| 2x2 stride,  outputs 5x5x8	padding Same	|
| Flatten				| output size = 200								|
| Fully connected		| size = 128,  RELU								|
| Fully connected		| size = 32,   RELU								|
| Fully connected (out)	| size = number of classes,   linear			|
 

#### 3. Model training

To train the model, I used a batch size of 128, 16 epochs, a learning rate of 0.00025 with an AdamOptimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.
My final model results were:
* training set accuracy of 98%
* validation set accuracy of 95% 
* test set accuracy of 93%

If an iterative approach was chosen:
* At first i chose an architecture consisting on a conv2d layer then a max_pool layer and then some fully connected layers because I tought it was a good starting point.
* I quickly achieved 85% accuracy with that architecture but it seemed that I needed more convolution and max pooling layers.
* After adding two other convolutions and max pooling layers I also added some dropout layers to help with overfitting.
* Lastly I tuned the learning rate which I found was too high at first because the accuracy was oscilating a lot.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


