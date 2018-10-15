
# coding: utf-8

# In[3]:


### Data pre-processing - See the report for detail
import csv
import os, os.path
import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

lines = []
# Standard csv reading
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
            
DIR = './IMG'
img_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

print (img_count, len(lines))
assert img_count/3 == len(lines)

# Data processing: Outward sample filtering
i=0
j=0
while j < len(lines):
    if float(lines[j][3])==0 and float(lines[j-1][3])==0:
        i+=1
    elif float(lines[j][3]) != 0:
        del lines[j-i//2:j]
        j-=i//2
        i=0

# Uncomment this condition if manual marking is used
#    if float(lines[j][7]) == 1:
#        del lines[j]
#        j-=1

    j+=1
    
# Random zero filtering
shuffle(lines)
new_lines=[]
angles=[]
for line in lines:
    if float(line[3])!=0 or random.randint(1, 40)==1:
        new_lines.append(line)
        angles.append(float(line[3]))
        angles.append(-float(line[3]))

plt.hist(np.asarray(angles, dtype='float'), bins=9)
plt.show()

# Uncomment this to check the pre-processing result in "pre-processed.csv" file
#with open('pre-processed.csv', 'w', newline='') as csvfile:
#    writer = csv.writer(csvfile)
#    for line in new_lines:
#        writer.writerow(line)

print("pre-processing done")
        
#for line in lines:
#    print(line[8])


# In[4]:


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from random import shuffle
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(new_lines, test_size=0.2)

print("Reading Finished")

# Generator with image flip
def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for row in batch_samples:
                angle_center = float(row[3])

                correction = 0.16
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                

                # split('\\') for windows, otherwise might want to change to split('/')
                f_center = './IMG/'+row[0].split('\\')[-1]
                f_left = './IMG/'+row[1].split('\\')[-1]
                f_right = './IMG/'+row[2].split('\\')[-1]
                img_center = cv2.imread(f_center)
                img_left = cv2.imread(f_left)
                img_right = cv2.imread(f_right)
                
                if img_center is not None and img_center.shape == (160,320,3):
                    img = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                    angle = angle_center
                    images.append(img)
                    angles.append(angle)
                
                    image_flipped = np.fliplr(img)
                    images.append(image_flipped)
                    angle_flipped = -angle
                    angles.append(angle_flipped)
 
                if img_left is not None and img_left.shape == (160,320,3):
                    img = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                    angle = angle_left
                    images.append(img)
                    angles.append(angle)
                
                    image_flipped = np.fliplr(img)
                    images.append(image_flipped)
                    angle_flipped = -angle
                    angles.append(angle_flipped)
                    
                if img_right is not None and img_right.shape == (160,320,3):
                    img = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                    angle = angle_right
                    images.append(img)
                    angles.append(angle)
                
                    image_flipped = np.fliplr(img)
                    images.append(image_flipped)
                    angle_flipped = -angle
                    angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

print("Data Generated")

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

import sys
print("python:",sys.version)
print("keras:",keras.__version__)

model = Sequential()
                   
# Pre-processing and image cropping
model.add(Lambda(lambda x: x/255.0-0.5, input_shape = (160,320,3)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2)))
#model.add(Cropping2D(cropping=((63,33),(0,0))))
model.add(Cropping2D(cropping=((11,5),(0,0))))


# Model Architecture - for two different versions

#for version 2.1.3
'''
model.add(Conv2D(24,(5,5), strides=(2,2), activation ='relu'))
model.add(Conv2D(36,(5,5), strides=(2,2), activation ='relu'))
model.add(Conv2D(48,(5,5), strides=(2,2), activation ='relu'))
model.add(Conv2D(64,(3,3), activation ='relu'))
model.add(Conv2D(64,(3,3), activation ='relu'))
model.add(Dropout(0.2))
'''
#for version 1.2.1

model.add(Conv2D(24,5,5,subsample=(2,2), activation = 'relu'))
model.add(Conv2D(36,5,5,subsample=(2,2), activation = 'relu'))
model.add(Conv2D(48,5,5,subsample=(2,2), activation = 'relu'))
model.add(Conv2D(64,3,3, activation = 'relu'))
model.add(Conv2D(64,3,3, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#model.load_weights('model20.h5')
print(len(train_samples), len(validation_samples))

'''
#for version 2.1.3
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= 6*len(train_samples),
                                     validation_data=validation_generator, shuffle = True,
                                     validation_steps=6*len(validation_samples), epochs=3, verbose = 1)
'''

#for version 1.2.1
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch = 6*len(train_samples), 
                                     validation_data = validation_generator,
                                     nb_val_samples=6*len(validation_samples), nb_epoch=3)


#Visualize loss and save model
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
print("done")

