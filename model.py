#first NN
import tensorflow as tf
from numpy import loadtxt,genfromtxt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, RNN, SimpleRNNCell, Activation, SimpleRNN, Lambda, Reshape
from tensorflow.keras.optimizers import Adam
import csv

#loading dataset contained in esc50.csv
with open("ESC-50/meta/esc50.csv") as f:
    reader = csv.reader(f)
    dataset = list(reader)

def make_spectrogram(signals):
    frame_length = tf.constant(2048)
    fft_length = tf.constant(512)
    spectrogram_complex = tf.signal.stft(signals, frame_length, fft_length)
    spectrogram = tf.math.abs(spectrogram_complex)
    #spectrogram = tf.transpose(spectrogram, perm=[0,2,1])
    tf.math.abs
    return spectrogram

#creates model that will be trained with all the layers
def create_model():
    conv_filters = 100
    conv_kernel_size = 100
    rnn_units = 100
    output_size = 1
    
    #forming model using tf.keras.Sequential()
    model = Sequential()
    #forming the input layer for model
    model.add(Input(shape=(None,), dtype=tf.float32))
    #forming the first lambda layer for model
    model.add(Lambda(lambda x:make_spectrogram(x)))
    #forming the first convolutional layer for model
    model.add(Conv1D(conv_filters, conv_kernel_size))
    #forming the simple RNN layer for model where output is fed back to input
    model.add(SimpleRNN(rnn_units))
    #forming the Dense layer for model 
    model.add(Dense(output_size))
    #forming the activation or output layer for model
    model.add(Activation('softmax'))
    
    #returning the final model with all the layers
    return model

#training the model to improve accuracy
def train_model(model,x_train,y_train):
    #learning rate(should decrease over time whilst training)
    lr = 1e-4
    #displays how quickly the learning rate is decreasing
    decay = 1e-6
    
    #sets up optimizer
    optim = Adam(lr=lr, decay=decay)
    #compiles the model showing loss and accuracy
    model.compile(optimizer=optim, loss = 'mse', metrics=['accuracy'])
    model.fit(x_train,y_train,epochs = 10)
#%%

import librosa as lib
import numpy as np

#audio_data = 'something.wav'
#x,sr = lib.load(audio_data)
#print(type(x), type(sr))
model = create_model()
#creating empty list to store the audiofile array
x_data = []
y_data = []
#this for loop stores each audio file in data set into an audio array and adds it to the list. Also, it checks if the category number matches 24, which is the category number for a cough.
for row in dataset[1:20]:
    print('I am a bad biyt')
    audio_data = row[0]
    full_path = "./ESC-50/audio/" + audio_data
    x,sr = lib.load(full_path)
    x_data.append(x)
    if row[3] == 24:
        y_data.append(1)
    else:
        y_data.append(0)
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

#Using scikitlearn to split into training and testing data.
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.10)
train_model(model,x_train,y_train)

model.save("./tf-model")

