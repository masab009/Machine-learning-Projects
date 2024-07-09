# hand-written-digits-with-CNNs
I used convolutional neural networks to predict the digit written in the image. I am getting started with computer vision and this was my first task after learning basics of computer vision and CNNs. 
Following architecture was used by me in this model
1. an Input layer of (m * 28 * 28 * 1) dimensions
2. a convolutional layer 1 with 3x3, 12 filters on the input images
3. Convolutional layer 2 with 5x5, 10 filters on the input from first layer
4. Flattening output of the conv2 layer for an output layer
5. Connecting the dense layer of 84 neurons was optional for me so, I commented it out
6. Conv 2 layer is connected to an output layer of 10 units, with a softmax classifier for assigning probabilities to the 10 classes 
*Notice that I personal took images of my handwritten digits, preprocessed them using open CV, made predictions on them and the model did a pretty fine job(classifying a single image incorrectly, there is a problem with my handwriting too:( )*
I found the last part to me most fun about the model :))) actually handwriting and then using the model to predict the digits. 
