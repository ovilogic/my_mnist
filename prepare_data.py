import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import matplotlib.pyplot as plt

# We will use 60,000 images to train the network and 10,000 images to evaluate how accurately 
# the network learned to classify images
'''The images are 28  x  28 arrays, with pixel values in the range [0, 255]. The labels are an array of integers, in the range [0, 9]. These correspond to the class of clothing the image represents:

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
            images = images.reshape((num_images, 28, 28))
            # 2. Normalize pixel values to the range [0, 1]
            images = images.astype(np.float32) / 255.0
            # 3. Reshape / add channel dimension
            #  Most CNN layers expect a 4D tensor (batch, height, width, channels).
            #  The reshape function requires integer dimensions.
            images = images.reshape((num_images, 28, 28, 1))
        return images
    
    def prep_labels(self):
        with open(self.labels_file, 'rb') as f:
            f.seek(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            # 4. Convert labels to one-hot encoding
            '''With one-hot encoding using np.eye(10)[labels], each label (an integer from 0 to 9) is 
            converted into a new array of length 10, where only the index corresponding to the original 
            label is 1 and all other indices are 0.'''
            labels = np.eye(10)[labels]  # Create a one-hot encoded array
        return labels
    
    def shuffle_ready(self):
        labels = self.prep_labels()
        images = self.prep_images()
        '''Shuffle the images and labels together to ensure that they correspond correctly after shuffling.'''
        indices = np.arange(images.shape[0]) # Create an array of indices from 0 to the number of images
        # 5. Shuffle the indices array
        # This ensures that the images and labels are shuffled in the same way.
        # The np.random.shuffle function shuffles the array in place.
        # This is important because we want to keep the correspondence between images and labels intact.
        # After shuffling, the images and labels will be in a random order, but each image will still correspond to its correct label.
        '''1. MNIST files store digits in label order (e.g., thousands of 0s, then 1s, then 2s...). If the model sees all the 0s before
         it sees any 1s, it can start to overfit to the early patterns, thinking they apply to everything.
        2. Ensures Diverse Mini-Batches
        If you train in mini-batches (e.g., batch_size=32), and you don't shuffle, 
        each batch might contain only a single class at first. That leads to poor gradient updates and slower learning.
        Example: First batch has all zeros → model thinks "everything must be zero!"
        Shuffling makes batches more representative of the overall data distribution.'''
        np.random.shuffle(indices)
        return images[indices], labels[indices]
    
extractor_train = DataPreparation(train_images_filename, train_labels_filename)

# Extract the first image and label for visualization
# ankle_boot_image = extractor_train.prep_images()[0]
# ankle_boot_image = ankle_boot_image.reshape((28, 28))  # Reshape to 2D for visualization
# ankle_boot_label = extractor_train.prep_labels()[0]
# # Save image
# plt.title('First MNIST Test Image')
# plt.axis('off')
# plt.imsave('ankleBoot_prepared.png', ankle_boot_image, cmap='gray')
# print(f'Ankle Boot Label: {ankle_boot_label}')

train_images, train_labels = extractor_train.shuffle_ready()

# print(f'Train images shape: {train_images.shape}, train labels shape: {train_labels.shape}')


extractor_test = DataPreparation(test_images_filename, test_labels_filename)

# Extract the first image and label for visualization from test data
# ankle_boot_test_image = extractor_test.prep_images()[0]
# ankle_boot_test_image = ankle_boot_test_image.reshape((28, 28))  # Reshape to 2D for visualization
# for i in range(ankle_boot_test_image.shape[0]):
#     for j in range(ankle_boot_test_image.shape[1]):
#         print(ankle_boot_test_image[i, j], sep=', ', end=' ')
# ankle_boot_test_label = extractor_test.prep_labels()[0]
# # Save image
# plt.title('First MNIST Test Image')
# plt.axis('off')
# plt.imsave('ankleBoot_prepared.png', ankle_boot_test_image, cmap='gray')
# print(f'Ankle Boot Label: {ankle_boot_test_label}')
# # Save the image array as a .csv file


# np.savetxt('ankle_boot_test_image.csv', ankle_boot_test_image, delimiter=',',
#             fmt='%f' # Use '%d' for integer format, or '%f' for float format if needed
#             )

test_images, test_labels = extractor_test.shuffle_ready()
print(test_images.shape, len(test_images[0][0]), test_labels[0], len(test_labels[0]), test_labels.shape)



                           