
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Flatten, Reshape, Activation
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy

from datagen import generate_data



def build_model() -> Model:
    # Input layer
    inputs = Input(shape=(9, 9, 10))
    
    # TimeDistributed layer to flatten each 3x3 box (9 squares * 10 values per square = 90 values in each box).
    x = TimeDistributed(Flatten())(inputs)

    # TimeDistributed applies the same series of Dense layer to each box, using the same weights and biases for each box.
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = TimeDistributed(Dense(32, activation='relu'))(x)
    
    # At this point, each 3x3 box which contained 90 values has been embedded/encoded as 32 values.
    # Flatten the 9 arrays of 32 values into a 1D array of 320 values.
    x = Flatten()(x)

    # Apply two Dense layers to the 320 values.
    x = Dense(384, activation='relu')(x)
    x = Dense(432, activation='relu')(x)

    # Reshape the 320 values into 9 arrays of 48 "encoded" values.
    x = Reshape((9, 48))(x)

    # TimeDistributed applies the same Dense layer to each of the 48 values in the 9 arrays, outputting an overall 9x81 array.
    x = TimeDistributed(Dense(81, activation='relu'))(x)
    
    # Reshape again.
    x = Reshape((9, 9, 9))(x)

    # Softmax activation is applied to each of the 9-element arrays that correspond to each cell, indicating which integer is most likely to be part of the solution to the puzzle.
    x = Activation('softmax')(x)
    
    # Output layer is the final value of x.
    outputs = x
    
    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[Accuracy()])
    
    return model



if __name__ == '__main__':
    model = build_model()
    model.summary()

    while True:
        X, Y = generate_data(16384)

        model.fit(X, Y, epochs=1, batch_size=128, validation_split=0.0625)
        model.save('timedist/sudoku.h5')

        del X, Y
