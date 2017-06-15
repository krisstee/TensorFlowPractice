#!/usr/bin/python

from keras import backend as K
from keras.layers import Input, Dense, Flatten, Activation, Dropout, Reshape
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D,merge
import keras.metrics
from keras.models import Sequential, Model
from keras.utils import np_utils
#from keras.utils.visualize_util import plot
import random
import numpy as np
#import cv2
from PIL import Image
import tensorflow as tf

WIDTH = 160 
HEIGHT = 60
CHANNEL = 3

# Optional - this will set so we can release GPU memory after computation
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

seed = 7
np.random.seed(seed)

def one_hot_encode (label) :
    return np_utils.to_categorical(np.int32(list(label)), 10)

# @NOTE: EDIT TO ORIGINAL CODE
def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    img_data = np.asarray(img,dtype=np.float32)
    return img_data

def load_data(path,train_ratio):
    datas = []
    labels = []
#    input_file = open(path + 'labels.txt')
    # @NOTE: EDIT TO ORIGINAL CODE
    input_file = open('captcha4_level1/labels.txt')
    for i,line in enumerate(input_file):
        img = Image.open(path + str(i) + ".png")
        data = img.resize([WIDTH,HEIGHT])
        data = np.multiply(data, 1/255.0)
        data = np.asarray(data)
        datas.append(data)
        labels.append(one_hot_encode(line.strip()))
        input_file.close()
        datas_labels = zip(datas,labels)
        # @NOTE: EDIT TO ORIGINAL CODE
        datas_labels = list(datas_labels)
        random.shuffle(datas_labels)
        (datas,labels) = zip(*datas_labels)
        size = len(labels)
        train_size = int(size * train_ratio)
        # @NOTE: EDIT TO ORIGINAL CODE
        datas_array = np.array(datas)
        labels_array = np.array(labels)
#        train_datas = np.stack(datas[ 0 : train_size ])
#        train_labels = np.stack(labels[ 0 : train_size ])
#        test_datas = np.stack(datas[ train_size : size ])
#        test_labels = np.stack(labels[ train_size : size])
        if(train_size == 0):
            train_datas = np.stack(datas_array)
            train_labels = np.stack(labels_array)
            test_datas = np.stack(datas_array)
            test_labels = np.stack(labels_array)
        else:
            train_datas = np.stack(datas_array[ 0 : train_size ])
            train_labels = np.stack(labels_array[ 0 : train_size ])
            test_datas = np.stack(datas_array[ train_size : size ])
            test_labels = np.stack(labels_array[ train_size : size])
        return (train_datas,train_labels,test_datas,test_labels)

def get_cnn_net():
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(HEIGHT, WIDTH, CHANNEL)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Flatten())
    x = model(inputs)
    print(inputs)
    x1 = Dense(10, activation='softmax')(x)
    x2 = Dense(10, activation='softmax')(x)
    x3 = Dense(10, activation='softmax')(x)
    x4 = Dense(10, activation='softmax')(x)
    x = merge([x1,x2,x3,x4],mode='concat')
    x = Reshape((4,10))(x)
    model = Model(input=inputs, output=x)

    #Visualize model
    #plot(model, show_shapes=True, to_file='simple_cnn_multi_output_model_captcha4.png')

    model.compile(loss='categorical_crossentropy', loss_weights=[1.], optimizer='Adam', metrics=['accuracy'])

    # other optimizer methods could be used : for example 
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #model.compile(loss='categorical_crossentropy', loss_weights=[1.], optimizer='sgd', metrics=['accuracy'])

    return model



(train_datas,train_labels,test_datas,test_labels) = load_data('captcha4_level1/',0.9)
model = get_cnn_net()
print(model)
model.fit(train_datas, train_labels, nb_epoch=32, batch_size=32, verbose=1)
predict_labels = model.predict(test_datas,batch_size=32)
test_size = len(test_labels)
y1 = test_labels[:,0,:].argmax(1) == predict_labels[:,0,:].argmax(1)
y2 = test_labels[:,1,:].argmax(1) == predict_labels[:,1,:].argmax(1)
y3 = test_labels[:,2,:].argmax(1) == predict_labels[:,2,:].argmax(1)
y4 = test_labels[:,3,:].argmax(1) == predict_labels[:,3,:].argmax(1)
acc = (y1 * y2 * y3 * y4).sum() * 1.0

print('\nmodel evaluate:\nacc:', acc/test_size)
print('y1',(y1.sum()) *1.0/test_size)
print('y2',(y2.sum()) *1.0/test_size)
print('y3',(y3.sum()) *1.0/test_size)
print('y4',(y4.sum()) *1.0/test_size)


# we have 10 sample untrained images to predict values for
# the following is an example to check how well your model can predict new un-seen information

for i in range(0,9):
#    chal_img = cv2.imread('captcha_data/level_1/captcha4_level1/'+ str(i) + ".png")
    chal_img = load_image('captcha_data/level_1/captcha4_level1/'+ str(i) + ".png")
#    resized_image = cv2.resize(chal_img, (WIDTH, HEIGHT)).astype(np.float32)
#    resized_image = np.expand_dims(resized_image, axis=0)
    # @NOTE: EDIT TO ORIGINAL CODE
    resized_image = np.resize(chal_img,(60,160,3))
    resized_image = np.expand_dims(resized_image, axis=0)
    out = model.predict(resized_image)		
    best_guess = np.argmax(out)

# logging to screen
# Best Guess the test data
print("Best Guess = ", np.argmax(out[:,0,:]),np.argmax(out[:,1,:]),np.argmax(out[:,2,:]),np.argmax(out[:,3,:]))

# Save model and weights for trained model
model.save_weights('captcha4_sample_model.h5')
with open('captcha4_sample_model.json', 'w') as f:
    f.write(model.to_json())


K.clear_session()
del sess
