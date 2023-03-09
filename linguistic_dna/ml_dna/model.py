import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.utils import to_categorical
from keras import Model, layers, Sequential, regularizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping

from preproc import *


def initialize_model(input_shape: tuple = (20,302,1)) -> Model:
    """
    Initialize the Neural Network
    """
    CNNmodel = Sequential()

    CNNmodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    CNNmodel.add(layers.MaxPooling2D((2, 2)))
    CNNmodel.add(layers.Dropout(0.2))
    CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
    CNNmodel.add(layers.MaxPooling2D((2, 2)))
    CNNmodel.add(layers.Dropout(0.2))
    CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
    CNNmodel.add(layers.Flatten())
    CNNmodel.add(layers.Dense(64, activation='relu'))
    CNNmodel.add(layers.Dropout(0.2))
    CNNmodel.add(layers.Dense(32, activation='relu'))
    CNNmodel.add(layers.Dense(5, activation='softmax'))

    print("✅ model initialized")
    return CNNmodel


def compile_model(CNNmodel: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    CNNmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    print("✅ model compiled")
    return CNNmodel


def train_model(CNNmodel: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=64,
                epochs=20,
                patience=2,
                validation_data=None, # overrides validation_split
                validation_split=0.2) -> tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=1)

    history = CNNmodel.fit(X,
                        y,
                        validation_data=validation_data,
                        validation_split=validation_split,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    print(f"✅ model trained")
    return CNNmodel, history


def evaluate_model(CNNmodel: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """
    if CNNmodel is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = CNNmodel.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True)

    accuracy = metrics['accuracy']

    print(f"✅ model evaluated: accuracy {round(accuracy, 2)}")
    return metrics


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
