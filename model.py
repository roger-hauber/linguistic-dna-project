# import necessary libraries
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Sequential, regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

# set file paths for CSV file and audio files
csv_path='/Users/frido/linguistic-dna-project/raw_data/train_twoaccents_large.csv'
audio_path='/Users/frido/linguistic-dna-project/raw_data/train_clips/'

# load data from CSV file into a pandas DataFrame
df = pd.read_csv(csv_path)

# delete rows in DataFrame that do not have corresponding audio files
audio_files = os.listdir(audio_path)
df_mini = df.loc[df["path"].isin(audio_files)]
df = df_mini
print(df.iloc[0])

# shuffle the rows of the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# add columns to the DataFrame with information about the audio files
#filelengths = [librosa.get_duration(path=file) for file in audio_path+df["path"]]
#df["length"] = filelengths
#df["num_words"] = [len(sent) for sent in df["sentence"].str.split()]
#df["num_words"].sum()/df["length"].sum()

# keep only the rows with accent values
#df_mod = df[~df["accent"].isna()]
df_mod=df

# define a function to trim or pad audio files to a specified length
def trim_pad_audio(file, cutoff=4, sr=22050):
    aud, sr = librosa.load(file)
    if aud.shape[0] < cutoff*sr:
        aud = np.pad(aud, pad_width=(0, (cutoff*sr)-aud.shape[0]))
        aud = aud[sr+1:]
        return aud
    aud = aud[sr+1:cutoff*sr,]
    return aud
print('audio trimmed')

# load audio files and trim/pad them to the specified length
aud_ser = [trim_pad_audio(file, cutoff=8) for file in audio_path + df_mod["path"]]
aud_ser_list = [file for file in audio_path + df_mod["path"]]
print(aud_ser_list[:10])
print(df_mod.iloc[:10])

#for file in audio_path + df_mod["path"]:
    



#
sr=22050


############################
#Choose Feature(s)
############################


# Get harmonic source of the audio file
def create_percussive(aud_ser):
    lst_harmonic = []
    lst_percussive = []
    for aud in aud_ser:
        y_harmonic, y_percussive = librosa.effects.hpss(aud)
        #lst_harmonic.append(y_harmonic)
        lst_percussive.append(y_percussive)
    print('split into harmonic and percussive source done')
    return lst_percussive

    

# compute MFCC features for the audio files and store them in a numpy array
def create_mfcc(aud_ser):
    lst_mfcc = []
    for aud in aud_ser:
        lst_mfcc.append(librosa.feature.mfcc(y=aud, n_mfcc=128))
    arr_mfcc = np.array(lst_mfcc)
    print('MFCC features computed')
    return arr_mfcc

# computeLog Mel-spectogram for the audio files and store them in a numpy array
def create_melspec(aud_ser):
    lst_melspec = []
    for aud in aud_ser:
        melspec = librosa.feature.melspectrogram(y=aud, sr=sr, n_mels=130)
        log_S = librosa.amplitude_to_db(melspec)
        lst_melspec.append(log_S)
    arr_melspec = np.array(lst_melspec)
    print('melspec features computed')
    return arr_melspec

# computeLog Chromagram for the audio files and store them in a numpy array
def create_chroma(aud_ser):
    lst_chroma = []
    for aud in aud_ser:
        lst_chroma.append(librosa.feature.chroma_cqt(y=aud, sr=sr))
        arr_chroma = np.array(lst_chroma)
    print('chroma features computed')
    return arr_chroma





# normalize the MFCC features
def scale_images(arr_mfcc):
    arr_mfcc_mmx = np.array((arr_mfcc-np.min(arr_mfcc))/(np.max(arr_mfcc)-np.min(arr_mfcc)))
    return arr_mfcc_mmx


from numpy import save

####Calling the functions - right now harmonic and chromogram
aud_ser=create_percussive(aud_ser)
aud_ser=create_chroma(aud_ser)
save('preproc_data.npy', aud_ser)
arr_mfcc_mmx=scale_images(aud_ser)
X_train=arr_mfcc_mmx
=======
# split the data into training and test sets
#X_train = arr_mfcc_mmx[:1800, :]
#X_test = arr_mfcc_mmx[1800:, :]
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
#X_train = np.expand_dims(X_train, axis=-1)
#X_test = np.expand_dims(X_test, axis=-1)

# split the data into training and test sets
#X_train = arr_mfcc_mmx[:1800, :]
#X_test = arr_mfcc_mmx[1800:, :]
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
X_train = np.expand_dims(X_train, axis=-1)
#X_test = np.expand_dims(X_test, axis=-1)
#print(X_train.shape)
#print(X_train[0][:])

# convert accent labels to categorical format
y_train = df_mod["accent"]
#y_test = df_mod["accent"]
y_train_cat = pd.get_dummies(y_train)
#y_test_cat = pd.get_dummies(y_test)
y_train_cat = np.array(y_train_cat)
#y_test_cat = np.array(y_test_cat)

# define a base model for a convolutional neural network
# input_shape=(20,302,1)
# CNNmodel = Sequential()
# CNNmodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# CNNmodel.add(layers.MaxPooling2D((2, 2)))
# CNNmodel.add(layers.Dropout(0.2))
# CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
# CNNmodel.add(layers.MaxPooling2D((2, 2)))
# CNNmodel.add(layers.Dropout(0.2))
# CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
# CNNmodel.add(layers.Flatten())
# CNNmodel.add(layers.Dense(64, activation='relu'))
# CNNmodel.add(layers.Dropout(0.2))
# CNNmodel.add(layers.Dense(32, activation='relu'))
# CNNmodel.add(layers.Dense(5, activation='softmax'))

# CNNmodel.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(learning_rate=0.1),

#               metrics=['accuracy'])

# history = CNNmodel.fit(X_train, y_train_cat, batch_size=16,  epochs=250, validation_split=0.3, shuffle=True)

#Model Victor
#def initialize_CNNmodel_1(X, y):

'''
First CNN Architecture, simple to avoid overfitting.
We should add up complication step-by-step.
'''
# For this dummy we have taken into account the mfccs size for height and width parameters
input_shape=(128,302,1)
CNNmodel = Sequential()
CNNmodel.add(layers.Conv2D(32, kernel_size=(11, 11), activation='relu', input_shape=input_shape))
CNNmodel.add(layers.MaxPooling2D(pool_size=(4, 4)))
CNNmodel.add(layers.Conv2D(16, kernel_size=(7, 7), activation='relu'))
CNNmodel.add(layers.MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(layers.Conv2D(8, kernel_size=(3, 3), activation='relu'))
CNNmodel.add(layers.MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(layers.Conv2D(4, kernel_size=(3, 3), activation='relu'))
CNNmodel.add(layers.MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(layers.Flatten())
CNNmodel.add(layers.Dense(32, activation='relu'))
CNNmodel.add(layers.Dropout(rate=0.2))
CNNmodel.add(layers.Dense(16, activation='relu'))
CNNmodel.add(layers.Dense(2, activation='softmax'))

CNNmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.1),
                             metrics=['accuracy'])

history = CNNmodel.fit(X_train, y_train_cat, batch_size=16,  epochs=50, validation_split=0.3, shuffle=True)

# evaluate model on test set                )
metrics = CNNmodel.evaluate(
        x=X_test,
        y=y_test_cat,
        batch_size=16,
        verbose=0,
        # callbacks=None,
        return_dict=True)

metrics['accuracy']


#Alternative Model
# model = Sequential()
# model.add(layers.Dense(100, activation='relu', input_shape=input_shape))
# model.add(layers.BatchNormalization(momentum=0.9))
# model.add(layers.Dropout(rate=0.1))
# model.add(layers.Dense(50, activation="relu"))
# model.add(layers.BatchNormalization(momentum=0.9))  # use momentum=0 to only use statistic of the last seen minibatch in inference mode ("short memory"). Use 1 to average statistics of all seen batch during training histories.
# model.add(layers.Dropout(rate=0.1))
# model.add(layers.Dense(5, activation="softmax"))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# history_2 = model.fit(X_train, y_train_cat, batch_size=16, validation_split=0.2)

# metrics = model.evaluate(
#         x=X_test,
#         y=y_test_cat,
#         batch_size=16,
#         verbose=0,
#         # callbacks=None,
#         return_dict=True)

print(metrics['accuracy'])


#load a model from a different project
# model_path='/Users/frido/linguistic-dna-project/models/copied_best_CNN.h5'
# model = keras.models.load_model(model_path)

# metrics = model.evaluate(
#         x=X_test,
#         y=y_test_cat,
#         batch_size=16,
#         verbose=0,
#         # callbacks=None,
#         return_dict=True)

# print(metrics['accuracy'])

from matplotlib import pyplot as plt
#history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.savefig('cnn_history.png')

CNNmodel.save("cnn_model.h5")
