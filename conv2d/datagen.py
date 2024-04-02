
# This script provides a function to generate a dataset for the convolutional neural network.

import numpy as np
import cv2



def generate_data(samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generate a dataset which maps images of triangles to the coordinates of their centroid.

    The bottom left corner of each image is at (-1, -1) and the top right corner is at (1, 1).

    Return a tuple containing the images and the vertices of the triangle that is in the corresponding image.
    '''
    # Initialize the array of images.
    X = np.zeros((samples, 24, 24, 1))

    # Initialize the array of centroid coordinates.
    Y = np.zeros((samples, 2))

    vertices = np.random.normal(0, .5, (samples, 3, 2))
    centroids = np.mean(vertices, axis=1)
    for i, (vs, centroid) in enumerate(zip(vertices, centroids)):
        # Generate a blank image.
        img = np.zeros((24, 24))

        # Draw the triangle on the image.
        cv2.fillPoly(img, [((vs + 1) * 12).astype(np.int32)], 1)

        # Record the image and centroid.
        X[i, :, :, 0] = img
        Y[i] = centroid

    return X, Y



if __name__ == '__main__':
    # Generate a dataset.
    X, Y = generate_data()

    # Display the each image scaled up by a factor of 16 along both dimensions.
    for img, centroid in zip(X, Y):
        print(centroid)
        cv2.imshow('Triangle', cv2.resize(img[::-1], (384, 384), interpolation=cv2.INTER_NEAREST))
        inp = cv2.waitKey(0)
        if inp & 0xff == 27: # ESC
            break
