from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2

# step 1: load data

img_width = 100
img_height = 80
train_data_dir = 'dataTrain'
valid_data_dir = 'dataValidation'
num_epocs =4

datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2, zoom_range=0.2)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
                                              target_size=(img_height, img_width),
                                              classes=['hand', 'other'],
                                              class_mode='binary',
                                              batch_size=16)

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
                                                   target_size=(img_height, img_width),
                                                   classes=['hand', 'other'],
                                                   class_mode='binary',
                                                   batch_size=32)


# step-2 : build model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=(img_height, img_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])


'''
model = load_model('models/handModel.h5')
print('model complied!!')
'''
print('starting training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=2048 //
                               16, epochs=num_epocs, validation_data=validation_generator, validation_steps=832//16)

print('training finished!!')

print('saving weights to simple_CNN.h5')

model.save_weights('models/simple_CNN.h5')
model.save('models/handModel.h5')

model_json = model.to_json()
with open("models/simple_CNN.json", "w") as json_file:
    json_file.write(model_json)

print('all weights saved successfully !!')
# models.load_weights('models/simple_CNN.h5')

'''
imageIndexes = [[4, 21, 24, 28, 34, 43], [15, 34, 56, 173, 181, 220]]

for i in  imageIndexes[0]:
    imagePath = "ds3/hand/hand (" + str(i) + ").png"
    img = cv2.imread(imagePath)
    testImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    testImage = cv2.resize(testImage, (60, 60))
    testImage = testImage.astype("float") / 255.0

    testImage = img_to_array(testImage)
    testImage = np.expand_dims(testImage, axis=0)
    # make predictions on the input image
    pred = model.predict(testImage)
    pred = pred[0][0]

    print(imagePath + " think its : " + str(pred));


for i in  imageIndexes[1]:
    imagePath = "ds3/other/nope (" + str(i) + ").png"
    img = cv2.imread(imagePath)
    testImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    testImage = cv2.resize(testImage, (60, 60))
    testImage = testImage.astype("float") / 255.0

    testImage = img_to_array(testImage)
    testImage = np.expand_dims(testImage, axis=0)
    # make predictions on the input image
    pred = model.predict(testImage)
    pred = pred[0][0]

    print(imagePath + " think its : " + str(pred));


'''