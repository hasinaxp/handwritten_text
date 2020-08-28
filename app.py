import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt


workspaceImage = np.zeros((128,128,3), np.uint8)
workspaceImageWindowName = "image"
workspaceImageShow = False
workspaceImageToolkitShow = False
imagePreviewScale = 600


def showImage(windowName, image):
    global imagePreviewScale
    h, w, nc = image.shape
    asp = w / h
    if asp > 1:
        w = imagePreviewScale
        h = w / asp
        print('hor')
    else:
        h = imagePreviewScale
        w = h * asp
        print('ver')
    temp = cv2.resize(image,(int(w), int(h)))
    cv2.imshow(windowName, temp)



def showWorkspaceImage(condition):
    global workspaceImage
    global workspaceImageWindowName
    if condition == True:
        showImage(workspaceImageWindowName, workspaceImage)
    else:
        cv2.destroyWindow(workspaceImageWindowName)


class RootWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        #title of the application
        self.title("HandWritten document selector")

        #menubar of the application
        self.mainMenuBar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.mainMenuBar, tearoff=0)
        self.fileMenu.add_command(label="Open doc image",command=self.openFile)
        self.fileMenu.add_command(label="Save doc image")
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Export marked items")
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit", command=self.destroy)
        self.mainMenuBar.add_cascade(label="File", menu=self.fileMenu)

        self.imageMenu = tk.Menu(self.mainMenuBar, tearoff=0)
        self.imageMenu.add_command(label="Show",
        command=lambda: showWorkspaceImage(True))

        self.imageMenu.add_command(label="Hide",
        command=lambda: showWorkspaceImage(False))
        self.imageMenu.add_separator()
        self.imageMenu.add_command(label="histogram",
        command=lambda: self.showhistogram())

        self.imageMenu.add_command(label="Detect Paragraph",
        command=lambda: self.showhistogram())

        self.imageMenu.add_command(label="Detect Word",
        command=lambda: self.showhistogram())

        self.mainMenuBar.add_cascade(label="Current Image", menu=self.imageMenu)

        self.config(menu=self.mainMenuBar)


        #widgets
        self.titleLabel = tk.Label(self, text="-modify image-").grid(row = 0, column = 0, columnspan=2 )
        self.rotationLabel = tk.Label(self, text="rotation :").grid(row = 1, column = 0 )
        self.rotationSlider = tk.Scale(self,length=200, from_=-180, to=180, orient=tk.HORIZONTAL)
        self.rotationSlider.grid(row = 1, column = 1)
        self.rmLineBtn = tk.Button(self, text="Remove Lines")
        self.rmLineBtn.grid(row = 2, column = 0 )
        self.clearBtn = tk.Button(self, text="Clear noise" )
        self.clearBtn.grid(row = 3, column = 0 )

        #currently working filepath
        self.inputImageFilePath = ""
    
    #menu events handlers
    def openFile(self):
        global workspaceImage
        global workspaceImageWindowName
        global workspaceImageShow

        self.inputImageFilePath = filedialog.askopenfilename(initialdir = "/",
        title = "Select image",
        filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
        workspaceImage = cv2.imread(self.inputImageFilePath)
        workspaceImageShow = True
        showWorkspaceImage(True)

    def showhistogram(self):
        global workspaceImage
        plt.hist(workspaceImage.ravel(),256,[0,256]); plt.show()

app = RootWindow()
app.mainloop()
