'''
##summary - 
    this file contains all image processing functions we required for
    for our final year project on document image processing to find 
    handwritten text segments

'''
#imports
import cv2
import numpy as np
import math
import os
from scipy.ndimage import interpolation as inter

#max dimention of the image
IMAGE_PREVIEW_SCALE = 800
#colors
COLOR_RED = (50, 80, 255)
COLOR_GREEN = (0, 255, 10)
COLOR_SKY = (255, 255, 30)
COLOR_BLUE = (255, 30, 0)

#function to show an image in  resized window
def showImage(windowName, image):
    global IMAGE_PREVIEW_SCALE
    h, w = image.shape[:2]
    asp = w / h
    if asp > 1:
        #print('horizontal document')
        w = IMAGE_PREVIEW_SCALE
        h = w / asp
    else:
        #print('vertical document')
        h = IMAGE_PREVIEW_SCALE
        w = h * asp
    temp = cv2.resize(image,(int(w), int(h)))
    cv2.imshow(windowName, temp)


#preprocess the image to remove excess information like noise
#and produce binary image
def clearImageToBinary(img):
    #shadow spot removal
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    pass2 = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
    #applying dialation and erosion to remove small noise
    kernel = np.ones((1,1), np.uint8)
    pass2 = cv2.dilate(pass2, kernel, iterations=1)
    pass2 = cv2.erode(pass2, kernel, iterations=1)
    #histogram adjustment
    histData = cv2.calcHist([pass2],[0],None,[256],[0,256])
    histBias = 140
    histBias2 = 180
    maxVal = 0
    maxInd = 0
    for i in range(80): 
        if histData[i] > maxVal:
            maxVal = histData[i]
            maxInd = i
    #print(maxInd)

    pass2[(pass2 < maxInd + histBias)] = 0
    pass2[(pass2 > histBias2)] = 255
    
    return pass2


#preprocess the image to correct the skew
def correctSkew(image, delta=1, limit=20):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = createKernelAniso(37, 15, 14)
    gray = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

#preprocess the image to remove excess information like noise
#and remove line segments from the image
def lineRemove(img):
    width, height, _ = img.shape
    result = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect lines
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(thresh,1,np.pi/180,200,minLineLength=100,maxLineGap=10)
    #remove lines with white color
    for line in lines:
        x1,y1,x2,y2 = line[0]
        xx = math.fabs(x1-x2)
        yy = math.fabs(y1-y2)
        if yy > 40 or xx > 100 :
            cv2.line(result,(x1,y1),(x2,y2),(255,255,255),4)

    return result

#remove line segments from the image
def clearLineRemove(img):
    width, height, _ = img.shape
    result = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect lines and remove using probabilistic hough transform
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(thresh,1,np.pi/180,200,minLineLength=100,maxLineGap=10)
    #remove lines with white color
    try:
       for line in lines:
        x1,y1,x2,y2 = line[0]
        xx = math.fabs(x1-x2)
        yy = math.fabs(y1-y2)
        if yy > 40 or xx > 100 :
            cv2.line(result,(x1,y1),(x2,y2),(255,255,255),6)
    except:
        pass
    
    rgb_planes = cv2.split(result)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result = cv2.merge(result_norm_planes)
    kernel = np.ones((1,1), np.uint8)
    result = cv2.dilate(result, kernel, iterations=1)
    result = cv2.erode(result, kernel, iterations=1)

    return result

#remobe lines from bin image
def lineRemoveBin(img):
    result = img.copy()
    #detect lines using probablistic  haugh transform 
    lines = cv2.HoughLinesP(img,1,np.pi/180,200,minLineLength=30,maxLineGap=6)
    #remove lines with white color
    for line in lines:
        x1,y1,x2,y2 = line[0]
        xx = math.fabs(x1-x2)
        yy = math.fabs(y1-y2)
        if yy > 40 or xx > 100 :
            cv2.line(result,(x1,y1),(x2,y2),(255,255,255),3)
    return result


# generate anisotropic kernel
#better to use for handwritten word detection where the patters are irregular
def createKernelAniso(kernelSize, sigma, theta):
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel



#find word contures inside image
def findContureData(img, minAreaFraction = 0.001):
    width, height  = img.shape
    minArea = minAreaFraction * width * height
    imgTemp = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # apply filter kernel
    kernel = createKernelAniso(37, 15, 14)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)

    i2 = cv2.erode(imgFiltered,(21,21),2)
    (_, _imgThres) = cv2.threshold(i2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - _imgThres
    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    j = 0
    hTotal = 0
    for c in components:
        # skip small word candidates
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        
        if w * h < minArea: #donot include conture if it is too small
            continue

        #calculating average height
        j += 1
        hTotal += h
        hAvrg  = hTotal / j

        if h > hAvrg * 1.8:
            h1 = int(h / 2)
            h2 = h - h1
            y1 = y + h1
            cbox1 = (x,y,w, h1)
            cbox2 = (x, y1, w, h2)
            res.append(cbox1)
            res.append(cbox2)
        else:
            res.append(currBox)

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry:entry[1] * width + entry[0])


#find line contures in paragraph image
def findContureDataLine(img):
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #turn the image into pieces of connected blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,6)) 
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    
    #finding contures
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # append components to result
    res = []
    j = 0
    for c in cnts:
        # skip small word candidates
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        res.append(currBox)

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry:entry[1])



#find paragraph contures of an image
def findContureDataParagraph(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    blur = cv2.GaussianBlur(img, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #turn the image into pieces of connected blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,6)) 
    dilate = cv2.dilate(thresh, kernel, iterations=6)
    
    #finding contures
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # append components to result
    res = []
    j = 0
    for c in cnts:
        # skip small word candidates
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        asp = w / h
       
        if asp > 2:  #if the image is wide too much cut into 2 pieces
            w1 = int(w / 2)
            w2 = w - w1
            x1 = x + w1
            cbox1 = (x,y,w1, h)
            cbox2 = (x1, y, w2, h)
            cimg1 = img[y:y+h, x:x+w1]
            cimg2 = img[y:y+h, x1:x1+w2]
            res.append((cbox1, cimg1))
            res.append((cbox2, cimg2))
        else:
            currImg = img[y:y+h, x:x+w]
            res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry:entry[0][1])

#remove extra white space
def trimWhiteSpace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    return rect

#mmark contours on an image
def drawContureRects(contures, image, color):
    for j, rect in enumerate(contures):
        (x, y, w, h) = rect
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)

#no longer needed
def saveParaContures(name, sliceFolder, contures):
    name = os.path.splitext(name)[0]
    for j,w in enumerate(contures):
        j = j + 1
        (wordBox, wordImg) =  w
        (x, y, w, h) = wordBox
        if w * h > 10000:
            #cv2.imwrite('%s/%s-%d.png'%(sliceFolder, imgName, j), wordImg)
            cv2.imwrite('%s/%s-%d.png'%(sliceFolder, name, j), wordImg)
    print("completed slicing " + name)
