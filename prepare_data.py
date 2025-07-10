import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import struct

# We will use 60,000 images to train the network and 10,000 images to evaluate how accurately 
# the network learned to classify images
'''The images are 28  ×  28 arrays, with pixel values in the range [0, 255]. The labels are an array of integers, in the range [0, 9]. These correspond to the class of clothing the image represents:

Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
Each image is mapped to a single label
The images are stored in a binary format, with the first 16 bytes containing metadata about the dataset, 
followed by the pixel values of the images. 
The labels are stored in a separate file, with the first 8 bytes containing 
metadata and the rest being the label values.
'''

train_images_filename = '/mnt/c/python_work/tensorFlow/wsl_venv/mnist/data_repo/extracted_data/' \
                        'train-images-idx3-ubyte'
test_images_filename = '/mnt/c/python_work/tensorFlow/wsl_venv/mnist/data_repo/extracted_data/' \
                       't10k-images-idx3-ubyte'
train_labels_filename = '/mnt/c/python_work/tensorFlow/wsl_venv/mnist/data_repo/extracted_data/' \
                        'train-labels-idx1-ubyte'
test_labels_filename = '/mnt/c/python_work/tensorFlow/wsl_venv/mnist/data_repo/extracted_data/' \
                       't10k-labels-idx1-ubyte'

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Meant to extract either training or testing data from the MNIST dataset.
# Size of images is 28x28 pixels, and labels are integers from 0 to 9. There isn’t a universal “standard” H×W that every CNN 
# must use—it’s really just that all your images must share the same dimensions when they enter the network.
class DataPreparation:
    def __init__(self, images_file, labels_file):
        self.images_file = images_file
        self.labels_file = labels_file

    def prep_images(self):
        with open(self.images_file, 'rb') as f:
            # 1. Extract images. The first 16 bytes contain metadata.
            f.seek(16)
            images = np.frombuffer(f.read(), dtype=np.uint8)
            num_images = images.size // (28 * 28)
            print(f'Number of images: {num_images}')
            images = images.reshape((num_images, 28, 28))
            # 2. Normalize pixel values to the range [0, 1]
            images = images.astype(np.float32) / 255.0
            # 3. Reshape / add channel dimension
            #  Most CNN layers expect a 4D tensor (batch, height, width, channels).
            #  The reshape function requires integer dimensions.
            images = images.reshape((num_images, 28, 28, 1))
        return images
    
    def extract_labels(self):
        with open(self.labels_file, 'rb') as f:
            f.seek(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            # 4. Convert labels to one-hot encoding
            '''With one-hot encoding using np.eye(10)[labels], each label (an integer from 0 to 9) is 
            converted into a new array of length 10, where only the index corresponding to the original 
            label is 1 and all other indices are 0.'''
            labels = np.eye(10)[labels]  # Create a one-hot encoded array
        return labels
    
extractor_train = DataPreparation(train_images_filename, train_labels_filename)
train_images = extractor_train.prep_images()
train_labels = extractor_train.extract_labels()
print(train_labels)

extractor_test = DataPreparation(test_images_filename, test_labels_filename)  
test_images = extractor_test.prep_images()
test_labels = extractor_test.extract_labels()
