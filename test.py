import cv2
import imagePreprocess as ipp #contains all functions for image processing
import arrangeText #contains fuunctions for aggregating result
import os
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

#importing neural network model
model = load_model('models/handModel2.h5')
print('model loaded!!')


'''
IMPORTANT

functions that contains X at the end save the data others simply show the data

bin - represents binary image. i.e. the images contain only one channel and has only 2 colors: black , white

'''

#returns float value that is handwritten if closest to 0
def predictHandwrittenParagraph(binImg):
    #converting image to be used by tenserflow
    testImage = cv2.cvtColor(binImg, cv2.COLOR_BGR2RGB)
    testImage = cv2.resize(testImage, (100, 80))
    testImage = testImage.astype("float") / 255.0

    testImage = img_to_array(testImage)
    testImage = np.expand_dims(testImage, axis=0)
    # make predictions on the input image
    pred = model.predict(testImage)
    return float(pred[0][0]) #return the prediction as fraction between 0 - 1



#test for predicting
#better version
def testPrediction2(folder, n1, n2):
    for testImagePath in os.listdir(folder)[n1:n2]:
        inputImage = cv2.imread(folder + '/' + testImagePath)
        width, height, _ = inputImage.shape
        lineFreeImage = ipp.clearLineRemove(inputImage)
        binImage = ipp.clearImageToBinary(lineFreeImage)
        ctours = ipp.findContureDataParagraph(binImage)
        positives = []
        blacklists = []
        for j, w in enumerate(ctours):
            (wordBox, wordImg) =  w
            (x, y, w, h) = wordBox
            pred = predictHandwrittenParagraph(wordImg)
            if pred < 0.5:
                positives.append(wordBox)
                #cv2.rectangle(lineFreeImage,(x,y),(x+w,y+h),ipp.COLOR_SKY,4)
            else:
                #cv2.rectangle(lineFreeImage,(x,y),(x+w,y+h),ipp.COLOR_RED,4)
                blacklists.append(wordBox)
        paraRects = arrangeText.arrangeParagraphs(positives,blacklists,width, height)
        for p in paraRects:
            cv2.rectangle(lineFreeImage,p[0], p[1],ipp.COLOR_GREEN,4)
        ipp.showImage(testImagePath, lineFreeImage)
    print('done')
    #ipp.showImage("original", inputImage)
    cv2.waitKey(0)

def testWordsDetection(img):
    binImage = ipp.clearImageToBinary(img)
    wordContures = ipp.findContureData(binImage)
    ipp.drawContureRects(wordContures, img, ipp.COLOR_BLUE)
    ipp.showImage("words", img)
    cv2.waitKey(0)

def testLineDetection(img):
    width, height, _ = img.shape
    binImage = ipp.clearImageToBinary(img)
    wordContures = ipp.findContureData(binImage)
    lineCtrs = arrangeText.arrangeLines(wordContures)
    for i, line in enumerate(lineCtrs):
        x1 = y1 = 9999
        x2 = y2 = 0
        for wb in line:
            if wb[0] < x1:
                x1 = wb[0]
            if wb[1] < y1:
                y1 = wb[1]
            if wb[0] + wb[2] > x2:
                x2 = wb[0] + wb[2]
            if wb[1] + wb[3] > y2:
                y2 = wb[1] + wb[3]

        cv2.rectangle(img,(x1,y1), (x2,y2),(i * 40 , 0,0),4)
    ipp.showImage("lines", img)
    cv2.waitKey(0)

def testLineDetection2(img):
    width, height, _ = img.shape
    binImage = ipp.clearImageToBinary(img)
    lineCtrs = ipp.findContureDataLine(binImage)
    for i, line in enumerate(lineCtrs):
        (x, y, w, h) = line
        cv2.rectangle(img,(x,y), (x + w,y+ h),(i * 40 , 0,0),4)
    ipp.showImage("lines", img)
    cv2.waitKey(0)



#test for predicting
def testPredictionX(folder,outFolder, n1, n2):
    for testImagePath in os.listdir(folder)[n1:n2]:
        inputImage = cv2.imread(folder + '/' + testImagePath)
        width, height, _ = inputImage.shape
        lineFreeImage = ipp.clearLineRemove(inputImage)
        binImage = ipp.clearImageToBinary(lineFreeImage)
        ctours = ipp.findContureDataParagraph(binImage)
        positives = [] #contains positive results
        blacklists = [] #contains negetive result
        for j, w in enumerate(ctours):
            (wordBox, wordImg) =  w
            (x, y, w, h) = wordBox
            pred = predictHandwrittenParagraph(wordImg)
            if pred < 0.5:
                positives.append(wordBox)
                #cv2.rectangle(lineFreeImage,(x,y),(x+w,y+h),ipp.COLOR_SKY,4)
            else:
                #cv2.rectangle(lineFreeImage,(x,y),(x+w,y+h),ipp.COLOR_RED,4)
                blacklists.append(wordBox)
        paraRects = arrangeText.arrangeParagraphs(positives,blacklists,width, height)
        for p in paraRects:
            cv2.rectangle(lineFreeImage,p[0], p[1],ipp.COLOR_GREEN,4)
        print('done processing ' + testImagePath)
        cv2.imwrite(outFolder + '/' + testImagePath, lineFreeImage)
    print('done')
    #ipp.showImage("original", inputImage)

def testWordsDetectionX(folder,outFolder, n1, n2):
    for testImagePath in os.listdir(folder)[n1:n2]:
        inputImage = cv2.imread(folder + '/' + testImagePath)
        width, height, _ = inputImage.shape
        binImage = ipp.clearImageToBinary(inputImage)
        wordContures = ipp.findContureData(binImage)
        ipp.drawContureRects(wordContures, inputImage, ipp.COLOR_BLUE)
        print('done processing ' + testImagePath)
        cv2.imwrite(outFolder + '/w_' + testImagePath, inputImage)

def testLineDetectionX(folder,outFolder, n1, n2):
    for testImagePath in os.listdir(folder)[n1:n2]:
        inputImage = cv2.imread(folder + '/' + testImagePath)
        width, height, _ = inputImage.shape
        binImage = ipp.clearImageToBinary(inputImage)
        wordContures = ipp.findContureData(binImage)
        lineCtrs = arrangeText.arrangeLines(wordContures)
        for i, line in enumerate(lineCtrs):
            x1 = y1 = 9999
            x2 = y2 = 0
            for wb in line:
                if wb[0] < x1:
                    x1 = wb[0]
                if wb[1] < y1:
                    y1 = wb[1]
                if wb[0] + wb[2] > x2:
                    x2 = wb[0] + wb[2]
                if wb[1] + wb[3] > y2:
                    y2 = wb[1] + wb[3]

            cv2.rectangle(inputImage,(x1,y1), (x2,y2),ipp.COLOR_RED,4)
        
        print('done processing ' + testImagePath)
        cv2.imwrite(outFolder + '/w_' + testImagePath, inputImage)


#functions for sliceing data into blocks
#used for preparing data for the training of neural network
def sliceData(folder, outputFolder):
    for testImagePath in os.listdir(folder):
        inputImage = cv2.imread(folder+'/' + testImagePath)
        lineFreeImage = ipp.lineRemove(inputImage)
        binImage = ipp.clearImageToBinary(lineFreeImage)
        ctours = ipp.findContureDataParagraph(binImage)
        ipp.saveParaContures(testImagePath, outputFolder, ctours)
        #ipp.showImage(testImagePath, lineFreeImage)
    print('done')



#crop handwritten part from original image and return array of paragraphs
def isolateHandwritten2(originalImage, contures, minAreaFraction = 0.02):
    width, height, _ = originalImage.shape
    minArea = width * height * minAreaFraction
    lineFreeImage = ipp.clearLineRemove(originalImage)
    binImage = ipp.clearImageToBinary(lineFreeImage)
    ctours = ipp.findContureDataParagraph(binImage)
    positives = []
    blacklists = []
    for j, w in enumerate(ctours):
        (wordBox, wordImg) =  w
        (x, y, w, h) = wordBox
        pred = predictHandwrittenParagraph(wordImg)
        if pred < 0.5:
            positives.append(wordBox)
        else:
            blacklists.append(wordBox)
    paraRects = arrangeText.arrangeParagraphs(positives,blacklists,width, height)
    outputs = []
    for p in paraRects:  
        (x1,y1) = p[0]
        (x2,y2) = p[1]
        w = x2 - x1
        h = y2 - y1
        if w * h < minArea:
            continue
        result = np.zeros([h, w,3],dtype=np.uint8)
        #print(p)
        #print(result.shape)
        result[0:h, 0:w] = originalImage[y1:y2, x1:x2]
        result = ipp.trimWhiteSpace(result)
        outputs.append(result)

    return outputs

#simple paragraph croping test
def testParaExtraction(inpImage):
    lineFreeImage = ipp.clearLineRemove(inpImage)
    binImage = ipp.clearImageToBinary(lineFreeImage)
    ctours = ipp.findContureDataParagraph(binImage)
    outs = isolateHandwritten2(lineFreeImage,ctours)
    for j, out in enumerate(outs):
    #out = ipp.currectSkew(out)
    #out = trimImage(out)
        ipp.showImage("x" + str(j), out)
    cv2.waitKey(0)


def cropParagraphsX(folder, outFolder, n1, n2):
    for testImagePath in os.listdir(folder)[n1:n2]:
        print("cropping paragraphs from " + testImagePath)
        inputImage = cv2.imread(folder + '/' + testImagePath)
        lineFreeImage = ipp.clearLineRemove(inputImage)
        binImage = ipp.clearImageToBinary(lineFreeImage)
        ctours = ipp.findContureDataParagraph(binImage)
        outs = isolateHandwritten2(lineFreeImage,ctours)
        for j, out in enumerate(outs):
        #out = ipp.currectSkew(out)
        #out = trimImage(out)
            cv2.imwrite(outFolder + "/x" + str(j) + testImagePath, out)
#testPrediction('../constain6',7,10)

def cropWordsX(folder, outFolder, n1, n2):
    for testImagePath in os.listdir(folder)[n1:n2]:
        print("cropping word from " + testImagePath)
        inputImage = cv2.imread(folder + '/' + testImagePath)
        width, height, _ = inputImage.shape
        binImage = ipp.clearImageToBinary(inputImage)
        ctours = ipp.findContureData(binImage)
        dirName = outFolder + '/' + testImagePath
        os.makedirs(dirName)
        for j, out in enumerate(ctours):
            (x, y, w, h) = out
            output = inputImage[y:y+h , x:x+w]
            cv2.imwrite(dirName + "/" + str(j)+'.tif', output)

