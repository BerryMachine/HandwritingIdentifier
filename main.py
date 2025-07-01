# Recreation of 3b1b handwritten digit identifier from scratch

import struct
import numpy as np

INPUT_SIZE = 784
H_SIZE = 128
OUTPUT_SIZE = 10

def read_images(file):
    with open(file, 'rb') as f:
        # Big-endian, 28x28 = 784, header 16 bytes; 4:filetype, 4:#imgs, 4:rows, 4:cols
        magic, n_imgs, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f'Invalid {magic} != 2051' # check file type

        img_data = np.frombuffer(f.read(), dtype=np.uint8) # convert to 0-255
        images = img_data.reshape(n_imgs, rows, cols)
        return images

def read_labels(file):
    with open(file, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        assert magic == 2049, f'Invalid {magic} != 2049'
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def relu(Z):
    return np.abs(Z)

def softmax(Z):
    ez = np.exp(Z - np.max(Z))
    return ez / (np.sum(ez, axis=0, keepdims=True))

def fw_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return A2

def bw_prop()
    
# He Initialization lololol good for ReLU
W1 = np.random.randn(H_SIZE, INPUT_SIZE) * np.sqrt(2 / INPUT_SIZE)
b1 = np.zeros((H_SIZE, 1))

# softer init for Softmax
W2 = np.random.randn(OUTPUT_SIZE, H_SIZE) * np.sqrt(1 / H_SIZE)
b2 = np.zeros((OUTPUT_SIZE, 1))