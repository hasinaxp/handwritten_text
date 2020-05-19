import imagePreprocess as ipp
import math
import cv2
import numpy as np

#return absolute angle between words in radians
def getTheta(wrd1, wrd2):
    x1 = wrd1[0]+wrd1[2] # x + w
    y1 = wrd1[1]
    x2 = wrd2[0]
    y2 = wrd2[1]
    #x2 should never be less than x1
    if x2 < x1:
        return (999, 9999)
    theta = math.fabs( math.atan2((y2 - y1),(x2 - x1)) )
    d = math.sqrt( (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1) )
    return (theta, d)

#greedy approach to arrange lines
#still under work not perfect yet
def arrangeLines(words, maxAngle = 0.2 , maxSpace = 300):
    lines = []
    line = []
    words = sorted(words, key=lambda entry:entry[1])
    cWord = False
    minX = 9999
    minY = 9999
    for wrd in words:
        if wrd[0] < minX and wrd[1] < minY:
            minX = wrd[0]
            minY = wrd[1]
            cWord = wrd
    
    while len(words) > 0:
        minD = 999
        found = False
        nextWord = False
        line.append(cWord)
        words.remove(cWord)
        for word in words:
            (t, d) = getTheta(cWord, word)
            if t < maxAngle and d < minD:
                minD = d
                found = True
                nextWord = word
        if found:
            cWord = nextWord
        else:
            lines.append(line)
            line = []
            minX = 9999
            minY = 9999
            for wrd in words:
                if wrd[0] < minX and wrd[1] < minY:
                    minX = wrd[0]
                    minY = wrd[1]
                    cWord = wrd

    final_lines = []
    
    while len(lines) > 1:
        fln = lines[0]
        lines.remove(fln)
        ymax = 0
        ymin = 9999
        for l in fln:
            if l[1] < ymin:
                ymin = l[1]
            if l[1] > ymax:
                ymax = l[1]
        canPush = True
        while canPush:
            canPush = False
            for ln in lines:
                if ln[0][1] <= ymax and ln[0][1] >= ymin:
                    canPush = True
                    for x in ln:
                        fln.append(x)
                    for l in fln:
                        if l[1] < ymin:
                            ymin = l[1]
                        if l[1] > ymax:
                            ymax = l[1]
                    lines.remove(ln)
        final_lines.append(fln)

    for l in final_lines:
        print(l)
        
    return final_lines


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
    

