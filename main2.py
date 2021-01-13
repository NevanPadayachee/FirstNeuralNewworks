import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#loads dataset
fashion_mnist = keras.datasets.fashion_mnist
#splits into training and testing
(train_images, train_labels), (test_images, test_labels)=fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model=keras.models.load_model('models/firstsave.h5')

COLOR = 'white'
plt.rcParams['text.color']=COLOR
plt.rcParams['axes.labelcolor']=COLOR

def get_number():
    while True:
        num = input("pick a number")
        if num.isdigit():
            num=int(num)
            if 0<=num<=1000:
                return num
            else :
                print("try again...")

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class=class_names[np.argmax(prediction)]
    show_image(image,class_names[correct_label],predicted_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()


num=get_number()
image=test_images[num]
label=test_labels[num]
predict(model, image, label)
print("Expected : "+class_names[test_labels[num]])
prediction = model.predict(test_images)
print("Guess : "+class_names[np.argmax(prediction[num])])