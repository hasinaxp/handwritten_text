# handwritten text region selection from document image

image processing project in python using opencv and keras.

## introduction

Document image analysis is promising area of research. Since two decades, it has attracted lot
of researchers to provide solutions to automation of document processing. Document image analysis
mainly deals with recognition and extraction of textual as well as graphical components. Text to speech,
signature verification, script identification, detection of name, address, pin-code etc., are the
applications of document image analysis. This proposed solution is to detect handwritten areas of a
document image and accruing it’s pixel data and dimensions so that the handwritten text can be
automated to further process for conversion into character strings as well as detection of the author.
The main workflow can be presented as the following:

> data -> pixel level processing ->contour detection -> feature level analysis


## execution
1. please makesure that python3 64bit is installed in the sysytem.
2. have tensorflow 1.x and opencv installed 

to install packages use the following commands:
>pip install tensorflow==1.13
>pip install opencv-python
>pip install scipy

-imagePreprocess.py contains all preprocessing functions
-handwrittenClassifierTrainer is used to train the neural network model
-test.py contains testcases for defferent procedures

## objective
Our endeavor is to automate the detection of the handwritten text so that features can be extracted from it without the help of any human agent.


## basic workflow-

- preprocessing the image for clearing the noise of paper and such as well as removing unnecessary lines.
- finding potential paragarph contures
- filering positive contures using a convolutional neural network
- arranging the positive results to detect paragraphs.
- finding word contures form found paragraphs by applying anisotropic kernals.
- arranging the word contures into potential lines using huristic search. (working on it)

## pixel level processing -
Clear functions present in `imagePreprocess.py` contains many pixel level preprocessing for enhancing the image before applying other algoriths. these makes use of other algorithms efficient but converts the image into black and white (single channel binary image).
these includes - 
- Shadow spot removal 
- adjustment of brightness 
- grey scale conversion 
- applying dilation and erosion to remove small noise
- histogram adjustment for binary image

## common issues-
- uneven distribution of text and presence of overlapping
- variation of handwritting
- variation of languages ( we are mainly testing on bengali, english and hindi samples)
- accuracy of cnn model for prediction
- sample data size
- font size and lighting conditions (not so important)



