#SoraniOCR.py

#A program to create a deep convolutional neural network
#to perform optical character recognition on Sorani Kurdish
#using TensorFlow, Keras, Pillow, Pydot, and Graphviz
#Made by Zachary Aaron Clark
#as part of a capstone project for a B.S. in Computer Science
#from SUNY Polytechnic Institute
#Last edit 4/29/2020

#The created model will achieve an average of approximately
#0.907 accuracy and 0.319 loss after 20 epochs of training
#Each epoch takes approx 6-10 seconds on a GTX 1080

#For directory management
import os
#For creation of the neural network
import keras
#To parse the data files and prepare them for training
from keras.preprocessing.image import ImageDataGenerator
#The model type being used
from keras.models import Sequential
#Different layers types that were tested or used in the final model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
#Different optimizers that were tested
from keras.optimizers import RMSprop, Adam, Adamax, Adadelta, SGD
#To create a visualization of the model
from keras.utils import plot_model

#Number of epochs to train for before testing
epochs = 20

#Create data generator
datagen = ImageDataGenerator()

#Load and iterate training dataset
train_it = datagen.flow_from_directory(os.path.dirname(__file__) + '/data/train/',
target_size=(32,32), class_mode='categorical', batch_size=50, color_mode='grayscale')
#Load and iterate validation dataset
val_it = datagen.flow_from_directory(os.path.dirname(__file__) + '/data/validation/',
target_size=(32,32), class_mode='categorical', batch_size=50, color_mode='grayscale')
#Load and iterate test dataset
test_it = datagen.flow_from_directory(os.path.dirname(__file__) + '/data/test/',
target_size=(32,32), class_mode='categorical', batch_size=50, color_mode='grayscale')

#Create model
model = Sequential()

#Input layer
model.add(Dense(64, activation='relu', input_shape=(32, 32, 1)))
model.add(Dropout(0.3))

#Baseline
'''
#Hidden layer 1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
'''

#Layer Type Testing
'''
#Dense Layers
#Hidden layer 1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Convolutional Layers
#Hidden layer 1
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Pooling Layers
#Hidden layer 1
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))
'''

#Biased Filtering Testing
'''
#Convolutional Layer with Bias
#Hidden layer 1
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu', use_bias=True))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu', use_bias=True))
model.add(Dropout(0.2))
'''

#Layer Amount Testing
'''
#Dense Layers
#Hidden layer 1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 3
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 4
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 5
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Convolutional Layers
#Hidden layer 1
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 3
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 4
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 5
model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Pooling Layers
#Hidden layer 1
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))

#Hidden layer 3
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))

#Hidden layer 4
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))

#Hidden layer 5
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))
'''

#Layer 'Size' Testing
'''
#Dense Layers
#Hidden layer 1
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.2))

#Convolutional Layers
#Filter Amount
#Hidden layer 1
model.add(Conv2D(2048, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Conv2D(2048, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Filter Size
#Hidden layer 1
model.add(Conv2D(512, kernel_size=(13, 13), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Conv2D(512, kernel_size=(13, 13), padding='valid', activation='relu'))
model.add(Dropout(0.2))

#Pooling Layers
#Hidden layer 1
model.add(MaxPooling2D(pool_size=(4, 4), padding='valid'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(MaxPooling2D(pool_size=(4, 4), padding='valid'))
model.add(Dropout(0.2))
'''

#Dropout Testing
'''
#Hidden layer 1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.99))

#Hidden layer 2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.99))
'''

#Activation Testing
'''
#Hidden layer 1
model.add(Dense(512, activation='softmax'))
model.add(Dropout(0.2))

#Hidden layer 2
model.add(Dense(512, activation='softmax'))
model.add(Dropout(0.2))
'''

#Final Model Layers
#Hidden layer 1
model.add(Conv2D(64, kernel_size=(9, 9), padding='valid', activation='relu'))
model.add(Dropout(0.3))

#Hidden layer 2
model.add(Conv2D(64, kernel_size=(9, 9), padding='valid', activation='relu'))
model.add(Dropout(0.3))

#Hidden layer 3
model.add(Conv2D(64, kernel_size=(9, 9), padding='valid', activation='relu'))
model.add(Dropout(0.3))

#Hidden layer 4
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))

#Output layer
model.add(Flatten())
#"19" represents the number of output categories. This number has been modified from 29
#for public release as 10 categories have data that is not available openly
model.add(Dense(19, activation='softmax'))

#Print a summary of the model to the screen
model.summary()

#Save a visualization of the model to a file
#plot_model(model, show_shapes=True, show_layer_names=False,
#			to_file='C:\ProgramData\Anaconda3\envs\capstone\capstone\model.png')

#Compile model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

#Train model
history = model.fit(train_it, epochs=epochs, verbose=1, validation_data=val_it)

#Test model
score = model.evaluate(test_it, verbose=0)
print('Test accuracy:', score[1])
print('Test loss:', score[0])