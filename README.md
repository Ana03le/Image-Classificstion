# Image-Classificstion using 2D Convolution Neuarl Network.
•	Implemented a convolutional neural network implementation on a non-trivial image processing. Developed neural network model which will identify whether a given image is the image of a cat or a dog.

•	Reference link for code:
https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
•	Link for data set used:
https://www.kaggle.com/c/dogs-vs-cats/data

•	Implementation Details:
Used 5000 images of cats and dogs for training neural network. Original data set contains 12500 images of each.
Four major steps for convolution neural networks are:
1.Convolution
2.Pooling
3.Flatteing
4.Full Connection

1.	Implemented model using keras packages. Initialized network as a sequential model.
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

2.	Imported keras Conv2D for performing convolution on training images of cats and dogs.

3.	The convolution layer computes the output of neurons that are connected to local regions or receptive fields in the input, each computing a dot product between their weights and a small receptive field to which they are connected to in the input volume. Each computation leads to extraction of a feature map from the input image. 
In the code:
image_classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
Generated convolution layer of 32 filters, with each of 3 * 3 matrix with “relu” activation function.
Input shape is RGB.

4.	Performed pooling operation in network by adding keras’s MaxPooling2D function layer to the network with matrix pool size of 2 * 2. Pooling operation helps in reducing size of the matrix.
image_classifier.add(MaxPooling2D(pool_size = (2, 2)))

Added two layers of convolution and MaxPooling.

5.	Have performed the flattening operation by adding flatten() layer to the network model. The process of flattening means converting the pooled images pixels input to a 2D array.
image_classifier.add(Flatten())

6.	Making the network model a fully connected network by adding below layers:
image_classifier.add(Dense(units = 128, activation = 'relu'))
image_classifier.add(Dense(units = 1, activation = 'sigmoid'))

7.	Performed ImageDataGenerator using keras’s ImageDataGenerator package.
Have rescaled the image data to values between 0 and 1 by diving each pixel by 255. As each pixel has maximum value is 255. Used ‘binary’ class mode and ‘binary_crossentropy’ loss function.
image_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

•	Model training:
The accuracy of model increased by increasing number of epochs and steps per epochs.
       Epoch= 10, steps_per_epoch = 80, loss: 0.3441 - acc: 0.8483
       Epoch= 20, steps_per_epoch = 800, loss: 0.1001 - acc: 0.9818
       Epoch= 25, steps_per_epoch = 8000, loss: 0.0231 - acc: 0.9999


