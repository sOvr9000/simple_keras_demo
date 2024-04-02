
# This script builds and trains a neural network to predict the centroid of a triangle given an image of the triangle.

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

from datagen import generate_data



def build_model() -> Model:
    '''
    Build a 2D convolutional neural network model.
    '''
    # Define the input layer.
    input_layer = Input(shape=(24, 24, 1))
    
    # Define the convolutional layers.
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Define the fully connected layers.
    flat = Flatten()(pool2)
    dense = Dense(128, activation='relu')(flat)
    output_layer = Dense(2)(dense)
    
    # Define the model.
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model



if __name__ == '__main__':
    # Build the model.
    model = build_model()
    
    # Compile the model.
    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Generate a dataset for training the model.
    train_X, train_Y = generate_data(samples=16384)
    
    # Train the model.
    model.fit(train_X, train_Y, epochs=10)
    
    # Save the model.
    model.save('triangle_centroid.h5')

