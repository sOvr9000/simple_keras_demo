
# This script loads a trained neural network and uses it to predict the centroid of a triangle given an image of the triangle.

import cv2

from keras.models import load_model

from datagen import generate_data



if __name__ == '__main__':
    # Load the model.
    model = load_model('triangle_centroid.h5')

    # Generate a small test dataset to view the image of the triangle, its actual centroid, and the predicted centroid.
    test_X, test_Y = generate_data(samples=16)

    pred = model(test_X).numpy()
    print(pred)

    for raw_img, actual, predicted in zip(test_X, test_Y, pred):
        print('Actual centroid:', actual)
        print('Predicted centroid:', predicted)

        # Resize image.
        img = cv2.resize(raw_img, (384, 384), interpolation=cv2.INTER_NEAREST)
        # Expand image to 3 channels.
        img = cv2.merge([img] * 3)
        # Draw medium size green circle around actual centroid.
        img = cv2.circle(img, tuple(map(int, (actual + 1) * 192)), 5, (0, 1, 0), -1)
        # Draw small size red circle around predicted centroid.
        img = cv2.circle(img, tuple(map(int, (predicted + 1) * 192)), 3, (0, 0, 1), -1)
        # Display image (flipped vertically).
        cv2.imshow('Triangle', img[::-1])

        inp = cv2.waitKey(0)
        if inp & 0xff == 27: # Press ESC to close the loop.
            break


