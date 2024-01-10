import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 

#Loading the dataset and splitting 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Data preprocessing
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

#Defining the stack of convolutional and pooling layers architecture 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Defining the dense layers architecture 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

#Training the model on the train data
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#Testing the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
