#import test
import cv2
import imagePreprocess as ipp

#test.sliceData('../constain6', 'samples')
fileName = 'folder1/folder2/0000_01_02.tif'

imagePath = 'Scan_30027.jpg'

img = cv2.imread(imagePath)
clearImage = ipp.clearImageToBinary(img)
cv2.imwrite("output.png", clearImage)

