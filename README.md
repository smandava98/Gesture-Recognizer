# Gesture-Recognizer
Static hand gesture recognition using vision based approach for human-computer interaction

Project Guide/Specifications:

In this project, my goal was to build a ML pipeline that recgonized the sign language alphabet just by looking at a person's hand through the webcam or computer camera. This type of recognition software, upon further improvement, can have many practical applications in medicine and technology ranging from helping stroke victims type on a keyboard to simply helping the deaf communicate more efficiently with other people around them. 

There are two key components to this project:

1. Build a static recognizer that predicts sign language
2. Locating the hand and feeding the image of the hand to the recognizer

* I am using pandas and scikit learn in this project for the machine learning. In terms of data, I am using a combination of Kaggle's MNIST Sign Language Dataset as well as an online dataset of people holding up different sign language symbols that was unfortunately too large to upload onto GitHub*

Building the Recgonizer Software:
Every image is converted into X,Y coordinates with the X representing the vector and Y representing the alphabet label. I used a histogram of oriented gradients feature extractor. Then, the Y values are transformed to numerical values. The model was then trained using SVC (Support Vector Classification) with linear kernel as opposed to SVC with RBF kernel which didn't work as well. Random Forest Classifiers are used to aid algorithmic performance observations. 

Building Hand and Training Classifier:
To check if it is a hand, I checked the degree of overlap with the given bounding box. To reduce false positives, hard negative mining is employed which eventually lead to a success rate of 82%. 

To summarize, given a test image, I first get the various detected regions across different scales of the image and pick the best one among them. This region is then cropped out, rescaled (to 128x128) and its corresponding hog vector is fed to the recognizer which then predicts the gesture denoted by the hand in the image.
