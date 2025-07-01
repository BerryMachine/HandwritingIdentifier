# Recreation of 3b1b handwritten digit identifier from scratch

import struct
import numpy as np

INPUT_SIZE = 784
H_SIZE = 128
OUTPUT_SIZE = 10
ITERS = 500
SCALE = 0.1

def read_images(file):
    with open(file, 'rb') as f:
        # Big-endian, 28x28 = 784, header 16 bytes; 4:filetype, 4:#imgs, 4:rows, 4:cols
        magic, n_imgs, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f'Invalid {magic} != 2051' # check file type

        img_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = img_data.reshape(n_imgs, rows * cols).T
        return images / 255.0 # normalize

def read_labels(file):
    with open(file, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        assert magic == 2049, f'Invalid {magic} != 2049'
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return np.eye(10)[labels].T

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    ez = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return ez / (np.sum(ez, axis=0, keepdims=True))

def fw_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def compute_loss(Y, A2):
    m = Y.shape[1]
    log_probs = np.log(A2 + 1e-8)
    return -np.sum(Y * log_probs) / m

def bw_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    dZ2 = A2 - Y
    dW2 = (1 / m) * (dZ2 @ A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = (Z1 > 0) * W2.T @ dZ2
    dW1 = (1 / m) * (dZ1 @ X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2
    
def gradient_descent(X, Y, iter, scale):
    # He Initialization good for ReLU heheehha
    W1 = np.random.randn(H_SIZE, INPUT_SIZE) * np.sqrt(2 / INPUT_SIZE)
    b1 = np.zeros((H_SIZE, 1))

    # Xavier Initialization for Softmax
    W2 = np.random.randn(OUTPUT_SIZE, H_SIZE) * np.sqrt(1 / H_SIZE)
    b2 = np.zeros((OUTPUT_SIZE, 1))

    for i in range(iter):
        Z1, A1, Z2, A2 = fw_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = bw_prop(Z1, A1, Z2, A2, W2, X, Y)
        
        # update
        W1 -= scale * dW1
        b1 -= scale * db1
        W2 -= scale * dW2
        b2 -= scale * db2

        if i and i % 100 == 0:
            print(f"{i}: loss = {compute_loss(Y, A2):.6f}")
        
        return W1, b1, W2, b2

# TRAIN
X = read_images("data/train-images.idx3-ubyte")
Y = read_labels("data/train-labels.idx1-ubyte")
W1, b1, W2, b2 = gradient_descent(X, Y, ITERS, SCALE)