# German Traffic Sign Classifier
Project 2 for Udacity' Self-Driving Car Nano Degree - Traffic Sign Classification

This project is my implementation of a classifier that can classify a subset of German Traffic Signs.

[//]: #(Image References)

[ image1 ]: ./sample_data/data_visualization.png "Visualization"
[ image2 ]: ./sample_data/bumpy_road.jpg "Traffic Sign 1"
[ image3 ]: ./sample_data/mandatory_roundabout.jpg "Traffic Sign 2"
[ image4 ]: ./sample_data/right_of_way.jpg "Traffic Sign 3"
[ image5 ]: ./sample_data/stop.jpg "Traffic Sign 4"
[ image6 ]: ./sample_data/traffic_signals.jpg "Traffic Sign 5" 

####1. Files Submitted
The following is a list of files included in my submission:

1) README.md - This file contains my writeup and analysis of my project.
2) Traffic_Sign_Classifier.ipynb - This is the Jupyter notebook that contains the Python code
3) Traffic_Sign_Classifier.html - This is an exported HTML file of the results from the Jupyter notebook.
4) bumpy_road.jpg, mandatory_roundabout.jpg, right_of_way.jpg, stop.jpg, traffic_signals.jpg - These are 5 new images of German Traffic Signs to test the accuracy of my classifier.  These are located in the "sample_data" folder.          

####2. Dataset Exploration 
I used the standard Python functions and Numpy to compute summary statistics of the traffic signs data set.  To acquire the number of classes, I created a set() data structure, since the set stores unique elements.

* The size of the training set is: 34799 Examples
* The size of the validation set is: 4410 Examples
* The size of the test set is: 12630 Examples
* The shape of a traffic sign image is: ( 32, 32, 3 )
* The number of unique classes/labels in the data set is: 43

I also outputted a sample of 10 random images from the training dataset.  These can be seen below:

![Sample Training Images][ image1 ]  

####3. Design and Test a Model Architecture

After visualizing several images from the dataset, I noticed that many signs had distinct colors (red, blue, etc.).  As a result, I decided against converting the images to grayscale.  I did perform image normalization on the training, validation, and testing images.  This normalization process helped remove certain noise, such as different intensities.  

The architecture of my Convolutional Neural Network is described as below:

Layer                         Description
:---------------------------------------:
Input                         32x32x3 RGB Image
Convolution 5x5               1x1 stride, valid padding, outputs 28x28x16
RELU
Max Pooling                   2x2 stride, outputs 14x14x16
Convolution 3x3               1x1 stride, valid padding, outputs 12x12x16
RELU
Convolution 3x3               1x1 stride, valid padding, outputs 10x10x16
RELU
Convolution 3x3               1x1 stride, valid padding, outputs 8x8x16
RELU
Dropout                       Keep Probability 0.5
Max Pooling                   2x2 stride, outputs 4x4x16
Flatten                       Input 4x4x16.  Flattens to output 256x1
Fully Connected               Input 256, outputs 240
RELU
Dropout                       Keep Probability 0.5
Fully Connected               Input 240, outputs 84
RELU
Dropout
Fully Connected               Input 84, outputs 43
Softmax

To train the model, I used the following parameters:
1) 10 Epochs
2) Batch Size of 256
3) Adam Optimizer
4) Learning Rate of 0.002

Originally, one of the first things I did was to increase the number of Epochs from 10 to 20 and 30.  I noticed that as my network architecture got better, adding extra epochs was frankly a waste of computing resources.  I would see the network's accuracy peak and the last epochs did not show any accuracy improvements.

I also doubled the learning rate from 0.001 to 0.002.  I did so because I valued a slow and steady learning rate, but also wanted the network to learn quicker.  Doubling the learning rate provided a nice compromise between stability and learning quicker. 

As one can see from the exported HTML, my final model results were:
* Training set: 99.1%
* Validation set: 96.2%
* Testing set: 95.5%

The training and validation percentages were calculated on a per-epoch basis.  For the testing set, the resulting network was saved, then loaded, and then the testing set was evaluated.

I originally started out with the LeNet architecture as I had finished the LeNet lab and decided to use it as a starting point.  After changing some of the parameters (mainly the depth to accomodate RGB colored images), I noticed that LeNet's accuracy did peak around the high-80% mark.  Additionally, I started seeing some overfitting occur on LeNet, namely I started having validation accuracy drop as the epochs increased.  From here I read some papers and presentations about AlexNet (the winner of the 2012 ILSVRC challenge) and Microsoft's winning submission for the 2015 ILSVRC challenge.  I borrowed different concepts from my findings.  First, I introduced dropout layers to help with the overfitting, starting with adding these layers between the fully-connected layers.  Just adding two dropout layers increased the accuracy of my network to the low-90's.  From the AlexNet architecture I borrowed the concept of chaining 3x3 Convolution layers with each other (please see layers 2-4) of my network.  I found that by increasing the depth of my network, my network became more accurate since there were more filters that can learn different image qualities.  I decided to chain 3x3 Convolutional layers together instead of also adding pooling layers between them since I wanted the network to learn image features on a large section of the image instead of a reduced one. 

I preserved the 2x2 size of the pooling layers, since halving the input layer was a large enough reduction. 

The greatest trade-off of adding more layers/filters is, of course, needing more epochs to tune all the parameters.  I played with filter depths of 16, 32, 48, and 64 and adding extra layers, and observed that in many instances I was not getting as much accuracy increases for the amount of time needed to train the network.  

####4. Test a model on New Images

These are five German traffic signs that I found on the Internet:

![Bumpy Road][ image2 ]
![Mandatory Roundabout][ image3 ]
![Right of Way][ image4 ]
![Stop][ image5 ]
![Traffic Signals][ image6 ]

For the first image (Bumpy Road), the dark background of the image may make this image difficult to classify.  If the classifier has been trained on bumpy road signs that have a lighter-colored background, the classifier may associate a light-colored background as a quality of a bumpy road sign.

For the second image (Mandatory Roundabout), this sign is very similar to other traffic signs, namely the "turn left ahead" and the "go straight or left" sign.  All three signs are circular, have a blue background, and have arrows pointing in a leftward direction.


In the third image (Right of Way), the background once again may be a problem.  The background contains parts of other signs (the right hand side of the image); therefore the classifier may expect a right-of-way sign to contain these portions.

The fourth image (Stop Sign) has an actual word in the image.  If the classifier is given a portion of a stop sign, it may not classify the image correctly as the classifier may expect the whole word "stop" to be present in the image.

Finally, the fifth image (Traffic Signals) is of very poor image quality; one can barely see the red and green dots (with the yellow almost whited out) on the image.  The classifier may classify a triangular shaped image without the dots as a traffic signals sign.

####Classifier's Predictions

As you can see from cells 8 and 9, my classifier correctly predicted all 5 images.  Using the labels from "signnames.csv", these are the results from the classifier:

Image                 Prediction  
---
Bumpy Road            (22) - Bumpy Road  
Mandatory Roundabout  (40) - Mandatory Roundabout  
Right of Way          (11) - Right of Way  
Stop                  (14) - Stop  
Traffic Signals       (26) - Traffic Signals  

This results in 100% accuracy for these images, which is even better than the 95.5% accuracy on the testing set.

####Softmax Predictions

The code for my softmax predictions is in cell 10.

My classifier was very certain about its predictions, as evidenced by the following tables:

Bumpy Road  
Probability          Prediction
---
0.99                 22 - Bumpy Road  
0.00005              26 - Traffic Signals  
0.00002              20 - Dangerous Curve to the Right  
0.00001              29 - Bicycles Crossing  
0.000005             25 - Road Work  

Mandatory Roundabout  
Probability         Prediction
---
0.915               40 - Mandatory Roundabout  
0.041               39 - Keep Left  
0.032               37 - Go straight or left  
0.002               11 - Right of way  
0.002               16 - Vehicles over 3.5 metric tons prohibited  

For the mandatory roundabout image, the classifier had a high confidence in the mandatory roundabout sign, but also gave (respectively) a 4% and 3% confidence to the two other signs that looked similar to it.

Right of Way  
Probability        Prediction
---
0.992               11 - Right of way  
0.0053              21 - Double curve  
0.002               30 - Beward of ice/snow  
0.0002              12 - Priority Road  
0.0001              27 - Pedestrians  
 
Stop  
Probability        Predictions
---
0.99                14 - Stop  
0.0003              17 - No entry  
0.00003              3 - Speed Limit (60km/h)  
0.00002             15 - No vehicles  
0.000008             9 - No passing  

Traffic Signals  
Probability        Predictions
---
0.99                26 - Traffic Signals  
0.0001              18 - General Caution  
0.000003            15 - No vehicles  
0.000002            20 - Dangerous curve to the right  
0.000001            12 - Priority Road  

As you can see the classifier had a very high confidence in its predictions.
