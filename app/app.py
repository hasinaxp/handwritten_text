import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import numpy as np
import cv2



#main window
class RootWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        #title of the application
        self.title("Handwritten Document selecter")
        #initial size of the application
        self.geometry("640x480+2+2")

        ####menubar of the application###
        self.mainMenuBar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.mainMenuBar, tearoff=0)
        self.config(menu=self.mainMenuBar)

        #file menu
        self.fileMenu = tk.Menu(self.mainMenuBar, tearoff=0)
        self.fileMenu.add_command(label="Open Files")
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit", command=self.destroy)
        self.mainMenuBar.add_cascade(label="File", menu=self.fileMenu)



app = RootWindow()
app.mainloop()
