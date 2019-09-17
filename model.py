# from google.colab import drive
# drive.mount('/content/drive')

# drive_path = '/content/drive/My Drive/Colab Notebooks/SDC_P4/'
drive_path = '/opt/carnd_p3/' # path on udacity workspace GPU mode

# !pip install keras==2.2.4
# !pip show keras  # 2.2.4 for Udacity workspace, 2.2.5 is default by Google Colab

# !pip show protobuf

import os
import csv
import cv2
import numpy as np
import pandas as pd

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D

"""### Prepare data set"""

df2 = pd.read_csv(drive_path + 'data/driving_log.csv')
# reduce similar training data, particularly to reduce steering angle =0 by a factor of 8
df3 = df2.loc[df2['steering']!=0]; 
df4 = df2.loc[df2['steering']==0][::8]; 
df = pd.concat([df3, df4])
imgName = df[['center', 'left', 'right']].values
angle = df['steering'].values

for i in range(0, len(imgName)):
  imgName[i][1] = imgName[i][1][1:]
  imgName[i][2] = imgName[i][2][1:]

# Split the data
X_trainfname, X_validfname, y_trainang, y_validang = train_test_split(imgName, angle, test_size=0.2, shuffle= True)
TrainSize = len(X_trainfname); 
ValidSize = len(X_validfname);


# print("Training Sample Size = ", TrainSize)
# print("Validation Sample Size = ",ValidSize)
# print("All Sample Size = ", len(df))

# # Visualize data size for triming
# hist, bins2 = np.histogram(angle, 21)
# print(hist)
# print(bins2)


# # need input matplot lib
# import matplotlib.pyplot as plt
# plt.hist(angle,bins = bins2)
# plt.show

"""### Data Generator"""

def aug_flip(images, angles):
    augflip_img = []
    augflip_ang = []
    for (img, ang) in zip(images, angles):
      augflip_img.append(img)
      augflip_ang.append(ang)
      augflip_img.append(np.fliplr(img))  #augflip_img.append(cv2.flip(img, 1))
      augflip_ang.append(ang*(-1.0))        
    return np.array(augflip_img), np.array(augflip_ang)

def generator(X_trainfname, y_trainang, batch_size=32):
    num_samples = len(y_trainang)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            batch_imgName = X_trainfname[offset:offset+batch_size]
            batch_angle = y_trainang[offset:offset+batch_size]
            images = []
            angles = []
            for k in range(0,len(batch_angle)):
              for j in range(0,3):
                name = drive_path + 'data/' + batch_imgName[k][j]

                image_bgr = cv2.imread(name)
                if image_bgr is None:
                  continue  
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                steer_angle = float(batch_angle[k])
                if j == 1:
                  steer_angle = steer_angle+0.2
                elif j ==2:
                  steer_angle = steer_angle-0.2

                  images.append(image_rgb)
                  angles.append(steer_angle)
            # call pre-process - flipping
            augflip_img, augflip_ang = aug_flip(images, angles)
            
          
            yield shuffle(np.array(augflip_img), np.array(augflip_ang))
            
train_generator= generator(X_trainfname, y_trainang)
validation_generator= generator(X_validfname, y_validang)


"""### Build Model"""

model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((60,25),(0,0)))) # 60 pixels from the top and 25 pixels from the bottom. ((top_crop, bottom_crop), (left_crop, right_crop))
model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

batch_size=32
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
            steps_per_epoch = TrainSize//batch_size, 
            validation_data = validation_generator, 
            validation_steps = ValidSize//batch_size, 
            epochs = 5, verbose = 1)
model.save('Build-model-10.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
# import matplotlib.pyplot as plt
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('Mean Squared Error Loss')
# plt.xlabel('Epoch Step')
# plt.legend(['Training set', 'Validation set'], loc='best')
# plt.show()

