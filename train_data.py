import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import tensorflow as tf
from prepare_data import train_images, train_labels, test_images, test_labels
from matplotlib import pyplot as plt

# Simlple ANN model for MNIST dataset
class SimpleANNModel:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            # In a fully connected (dense) layer, each neuron expects a one-dimensional vector of inputs. 
            # If your raw data comes in as a multi-dimensional tensor—say, a 28×28 pixel image—you need 
            # to linearize it so that each pixel becomes one entry in that vector. The Flatten operation 
            # simply reshapes (batch_size, height, width, channels) 
            # into (batch_size, height × width × channels) before feeding it into the dense layer.
            # If your network begins with one or more Dense layers and your input isn’t already a vector, you must flatten
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_images, train_labels, epochs=5):
        self.model.fit(train_images, train_labels, epochs=epochs)

    def evaluate(self, test_images, test_labels):
        return self.model.evaluate(test_images, test_labels)

    
compiled_model = SimpleANNModel()
trained_model = compiled_model.train(train_images, train_labels)
'''The 313/313 in the output of evaluate() refers to the number of batches processed during evaluation.

313 is the total number of batches needed to go through your entire test set.
By default, Keras uses a batch size of 32 for evaluation.
The MNIST test set has 10,000 images.
10,000 / 32 ≈ 312.5, which rounds up to 313 batches.
So, Keras processed your test data in 313 batches of up to 32 images each.'''
# tested_model = compiled_model.evaluate(test_images, test_labels)

first_image = test_images[0:1]
prediction1 = compiled_model.model.predict(first_image)
plt.title('first prediction')
first_image = first_image.reshape((28, 28))
plt.imsave('first_prediction.png', first_image, cmap='gray')
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
When you see something like 6.0925932e-01, that is scientific notation for:
6.0925932 x 10^(-1) = 0.60925932
'''
predicted_label = np.argmax(prediction1, axis=1)[0]
rounded_prediction = np.round(prediction1, 2)
print(rounded_prediction)
print("Predicted label:", predicted_label, "which is a", class_names[predicted_label])

first_label = test_labels[0]
if first_label == predicted_label:
    print("The prediction is correct!")
else:
    print("The prediction is incorrect. The correct label is:", class_names[first_label],
          "what I got is:", class_names[predicted_label])