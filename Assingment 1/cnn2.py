
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime



from tensorflow import keras

num_classes = 10


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


#load cifar10 data set 

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

#cnormalize the images

train_images, test_images = train_images/255.0, test_images/255.0

#assign labels categories

target_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# We use the Sequential model in keras which is used 99% of the time 

model = tf.keras.Sequential() 

# We add our first convolutional layer with 64 neurons and filter size of 3 x 3 

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape= (32,32,3)))

# We add our max pooling layer 

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#We add a second convolutional layer 

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu')) 


#add second  max pooling layer

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))


# Flattern the layer to a vector

model.add(tf.keras.layers.Flatten())

# A fully connected layer

model.add(tf.keras.layers.Dense(64, activation='relu')) 





# We add an output layer that uses softmax activation for the 10 classes

model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))



model.summary()

#we compile our model and add a loss function along with an optimization function.

model.compile(optimizer= 'adam', loss=tf.keras.losses.sparse_categorical_crossentropy,  metrics=['accuracy'])



#Next, we train our model based on the training set and test set

history=model.fit(train_images, train_labels, batch_size=128, epochs=3, verbose=1)

#plt training accuracy

plt.plot(history.history['accuracy'], label=['Accuracy'])
plt.plot(history.history['loss'],label =['Loss'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy/ Loss')
plt.ylim([0.5, 2])
plt.legend(loc='upper left')
plt.show()


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Accuracy',test_acc)




prediction  = model.predict (test_images)

i=20
print (prediction[i])
print(test_labels[i])
t=test_labels[i]
print ('t', t)
print (t[0])
print ("tested", target_labels[t[0]])
print(np.argmax(prediction[i]))
print ("predicted" , target_labels[np.argmax(prediction[i])])

for i in range(5):

    plt.grid(False)
    plt.figure(figsize= (4,4))
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    t=test_labels[i]
  
    plt.xlabel("Actual : " + target_labels[t[0]])
 
    plt.title("Predicted :" +  target_labels[np.argmax(prediction[i])])
plt.show()








