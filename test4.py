import test
import cv2

#test.testPrediction('../constain6', 'paraScreenShot', 50, 120);

#img = cv2.imread('dataTemp/Scan_10004.jpg')
#test.testParaExtraction(img)

#test.cropParagraphs('../constain6', 'paragraphs',50, 150)

#img = cv2.imread('paragraphs/x0Scan0011.jpg')

img = cv2.imread('paragraphs/x0Scan_10012.jpg')
test.testLineDetectionX('paragraphs', 'lineScreenShot',1, 10)