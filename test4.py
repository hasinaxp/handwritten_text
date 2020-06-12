import test
import cv2
import os


#test.testPrediction('../constain6', 'paraScreenShot', 50, 120);

#img = cv2.imread('dataTemp/Scan_10004.jpg')
#test.testParaExtraction(img)

#test.cropParagraphs('../constain6', 'paragraphs',50, 150)

#img = cv2.imread('paragraphs/x0Scan0011.jpg')


folder = 'paragraphs'
files = os.listdir(folder)

index = 1
while(index < 7):
    imgPath = folder + '/' + files[index] 
    img = cv2.imread(imgPath)
    test.testLineDetection2(img)
    index = index + 1
