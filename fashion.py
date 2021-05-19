from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data. load_data()

class_names = ['T shirt','Trouser', 'Pull over', 'Dress', 'Coat','Sandal', 
'Shirt', 'Sneaker', 'Bag', 'Ancle_boot']


train_images = train_images/255.0
test_images = test_images/255.0


#Build model
model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten (input_shape = (28,28)))
model.add(tf.keras.layers.Dense (128,activation="relu"))
model.add(tf.keras.layers.Dense(10, activation ="softmax"))
	                     
model.compile (optimizer= "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

#training

model.fit(train_images,train_labels,epochs=10)	

#testing

test_loss, test_acc = model.evaluate(test_images, test_labels) 
#print ('Test Loss', test_loss)   
print ('Test Accuracy', test_acc)

model.save('my_model.h5')
#classification 


prediction  = model.predict (test_images)

p= prediction[10]
print(p)
print(np.argmax(p))
print(class_names[test_labels[10]])
print(class_names[np.argmax(p)])


for i in range(5):

  plt.grid(False)
  plt.imshow(test_images[i],cmap=plt.cm.binary)
  
  plt.xlabel("Actual : " + class_names[test_labels[i]])
 
  plt.title("Predicted :" +  class_names[np.argmax(prediction[i])])
  plt.show()






