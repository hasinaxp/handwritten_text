import cv2
import imagePreprocess as ipp
import os
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


#returns float value that is handwritten if closest to 0
def predictHandwrittenParagraph(binImg, model):
    testImage = cv2.cvtColor(binImg, cv2.COLOR_BGR2RGB)
    testImage = cv2.resize(testImage, (100, 80))
    testImage = testImage.astype("float") / 255.0

    testImage = img_to_array(testImage)
    testImage = np.expand_dims(testImage, axis=0)
    # make predictions on the input image
    pred = model.predict(testImage)
    return float(pred[0][0])

#crop handwritten part from original image
def isolateHandwritten(originalImage, contures, model):
    height, width = originalImage.shape[:2]
    result = np.zeros([height,width,3],dtype=np.uint8)
    result.fill(255) # or img[:] = 255
    
    xx1 = width
    yy1 = height

    xx2 = 0
    yy2 = 0
   
    for j, w in enumerate(contures):
        (wordBox, wordImg) =  w
        (x, y, w, h) = wordBox
        pred = predictHandwrittenParagraph(wordImg,model)
        #print('prediction -> ' + str(pred))
        if pred < 0.5: #potentially handwritten
            result[y:y+h, x:x+w] = originalImage[y:y+h, x:x+w]
            if x < xx1: xx1 = x
            if y < yy1: yy1 = y
            if x+w > xx2: xx2 = x+w
            if y+h > yy2: yy2 = y+h
    output = np.zeros([yy2 - yy1, xx2 - xx1, 3],dtype=np.uint8)
    output[0:yy2, 0:xx2] = result[yy1:yy2, xx1:xx2]

    
    return output

