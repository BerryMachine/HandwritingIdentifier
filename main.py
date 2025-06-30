# Recreation of 3b1b handwritten digit identifier from scratch

import struct
import numpy as np

def read_images(file):
    # Open in binary
    with open(file, 'rb') as f:
        # Big-endian, 28x28 = 784, header 16 bytes; 4:filetype, 4:#imgs, 4:rows, 4:cols
        magic, n_imgs, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f'Invalid {magic}' # check file type

        img_data = np.frombuffer(f.read(), dtype=np.uint8) # convert to 0-255
        images = img_data.reshape(n_imgs, rows, cols)
        return images

