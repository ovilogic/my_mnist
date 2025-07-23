import numpy as np
import struct
import matplotlib.pyplot as plt

packed = struct.pack('>I', 1234)
print(packed, type(packed))

unpacked = struct.unpack('>I', packed)
print(unpacked, type(unpacked))

filename = '/mnt/c/python_work/tensorFlow/wsl_venv/mnist/data_repo/extracted_data/' \
't10k-images-idx3-ubyte'

with open(filename, 'rb') as f:
    magic_number = struct.unpack('>I', f.read(4))[0]
    print(f'Magic number: {magic_number}')

    num_images = struct.unpack('>I', f.read(4))[0]
    print(f'Number of images: {num_images}')

    num_rows = struct.unpack('>I', f.read(4))[0]
    print(f'Number of rows: {num_rows}')

    num_cols = struct.unpack('>I', f.read(4))[0]
    print(f'Number of columns: {num_cols}')

    image_size = num_rows * num_cols
    print(f'Image size: {image_size}')
    first_image = f.read(image_size)
    image = np.frombuffer(first_image, dtype=np.uint8).reshape((num_rows, num_cols))
    
    # Save image
    plt.title('First MNIST Test Image')
    plt.axis('off')
    plt.imsave('first_mnist_test_image.png', image, cmap='gray')

    # all_images = np.array([])

    # all_debuffered_images = np.frombuffer(f.read(-1), dtype=np.uint8)
    # print(f'All debuffered images shape: {all_debuffered_images.shape}')

    # num_images = all_debuffered_images.size // 784
    # print(num_images)

    # all_images = all_debuffered_images.reshape((num_images, num_rows, num_cols))
    # print(all_images[0].shape, all_images[0])

    

    # de_buffered = np.frombuffer(first_image, dtype=np.uint8)
    # print(de_buffered)
    # reshaped_image = de_buffered.reshape((num_rows, num_cols))
    # print(reshaped_image.shape, reshaped_image)

