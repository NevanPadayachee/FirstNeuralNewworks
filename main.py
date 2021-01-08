import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#loads dataset
fashion_mnist = keras.datasets.fashion_mnist
#splits into training and testing
(train_images, train_labels), (test_images, test_labels)=fashion_mnist.load_data()

#print(train_images.shape)

#looking at the greyscale of one pixel [image_number,pixel,pixel]
#print(train_images[0, 23, 23])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#looking at one of the images
#plt.figure()
#plt.imshow(train_images[8])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#layer1 injput
    keras.layers.Dense(128, activation='relu'),#layer2 hidden
    keras.layers.Dense(10, activation='softmax')#layer3 output
])

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc= model.evaluate(test_images,test_labels, verbose=1)

print('Test accuracy: ', test_acc)

prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[8])])
plt.figure()
plt.imshow(test_images[8])
plt.colorbar()
plt.grid(False)
plt.show()