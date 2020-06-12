import imagePreprocess as ipp
import math
import cv2
import numpy as np

def arrangeWords(words, width, height, factor = 1.2):
    blockHeight = words[0][3] * factor
    initHeight = words[0][1]
    sortedWords = sorted(words, key=lambda entry:int((entry[1] + entry[3]/2)/ blockHeight) * width + entry[0])
    return sortedWords




#function to arrange contures into paragraph
#finds clusters of positive feed and arranges to find the dimentions of the paragraph
def arrangeParagraphs(ctours, negetives, width,  height, space = 0.09, nspace = 0.01, minAreaFrac = 0.01):
    paragraphs = []
    wspace = height * space
    wnspace = height * nspace
    minArea = minAreaFrac * width * height
    currentHeight = 0
    previousHeight  = 0
    currentSegments = []
    for j, ctr in enumerate(ctours):
        (x, y, w, h) =  ctr
        if math.fabs(y - currentHeight) > wspace and math.fabs(y - previousHeight) > wspace:
            paragraphs.append(currentSegments)
            currentSegments = []
        canPush = True
        if len(currentSegments) < 3:
            for neg in negetives:
                if math.fabs(y - neg[1]) < wnspace:
                    canPush = False
        if canPush:
            currentSegments.append(ctr)
            currentHeight = y + h
            previousHeight = y

    paragraphs.append(currentSegments)
    paragraphRects = []
    for p in paragraphs:
        if len(p) > 0:
            x1 = y1 = 9999
            x2 = y2 = 0
            for wb in p:
                if wb[0] < x1:
                    x1 = wb[0]
                if wb[1] < y1:
                    y1 = wb[1]
                if wb[0] + wb[2] > x2:
                    x2 = wb[0] + wb[2]
                if wb[1] + wb[3] > y2:
                    y2 = wb[1] + wb[3]
            
            if (x2 - x1) * (y2 - y1) > minArea:
                paragraphRects.append(((x1,y1),(x2,y2)))
    #print(paragraphRects)
    return paragraphRects
    

