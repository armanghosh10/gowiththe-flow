import tensorflow as tf
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

#Loading the dataset
fashion_mnist = keras.datasets.fashion_mnist 
#Splitting into test data and training data 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 

class_names = ['T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Data pre-processing 
train_images = train_images / 255.0 
test_images = test_images / 255.0

#Defining the model architecture
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#Training the model on the training data
model.fit(train_images, train_labels, epochs=10)

#Testing the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy: ', test_acc)

#Comparing the model predictions and accuracy of the model
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
