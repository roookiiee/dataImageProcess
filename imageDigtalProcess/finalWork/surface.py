import getpass
import os
import winreg
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

import cv2 as cv
from PIL import Image, ImageTk

import vehicleLicense
import time


# PicPath: Save the last picture file path, this picture should be opened successful.
class LPRPath:
    def getPath(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\{}\LPR".format(getpass.getuser()))
            self.path = winreg.QueryValueEx(key, "LPR")
        except:
            self.path = None
        return self.path

    def setPath(self, path):
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\{}\LPR".format(getpass.getuser()))
        winreg.SetValueEx(key, "LPR", 0, winreg.REG_SZ, path)
        self.path = path


# Main LPR surface
class LPRSurface(Tk):
    labelPicWidth = 700
    labelPicHeight = 700
    buttonWidth = 100
    buttonHeight = 50
    textWidth = 10
    textHeight = 50
    tkWidth = labelPicWidth
    tkHeigth = labelPicHeight + buttonHeight * 4
    isPicProcessing = False
    root = None
    entryPlateNum = None
    entryPlateColor = None

    def resizePicture(self, imgCV):
        if imgCV is None:
            print("Read Fail!")
            return None

        imgCVRGB = cv.cvtColor(imgCV, cv.COLOR_BGR2RGB)
        img = Image.fromarray(imgCVRGB)
        imgTK = ImageTk.PhotoImage(image=img)

        picWidth = imgTK.width()
        picHeight = imgTK.height()
        # print("Picture Size:", picWidth, picHeight)
        if picWidth <= self.labelPicWidth and picHeight <= self.labelPicHeight:
            return imgTK

        widthScale = 1.0 * self.labelPicWidth / picWidth
        heightScale = 1.0 * self.labelPicHeight / picHeight

        scale = min(widthScale, heightScale)

        resizeWidth = int(picWidth * scale)
        resizeHeight = int(picHeight * scale)

        img = img.resize((resizeWidth, resizeHeight), )
        imgTK = ImageTk.PhotoImage(image=img)

        return imgTK

    # Load picture
    def loadPicture(self):
        start = time.time()
        # Get Picture Path
        if True == self.isPicProcessing:
            print("Please wait until previous picture process finish!!!")
            messagebox.showerror(title="PROCESSING", message="Please wait until previous picture process finish!!!")
            return
        self.videoThreadRun = False

        LPRPic = LPRPath()
        if None == LPRPic.getPath():
            initPath = ""
        else:
            initPath = LPRPic.path

        # fileName = None
        fileName = filedialog.askopenfilename(title='Load Picture', \
                                              filetypes=[('Picture File', '*.jfif *.jpg *.png *.gif'),
                                                         ('All Files', '*')], \
                                              initialdir=initPath)

        print(fileName)
        if not os.path.isfile(fileName):
            print("Please input correct filename!")
            return False
        # Read Picture File.
        try:
            # self.imgOri = Image.open(fileName)
            # imgCV = cv.imdecode(np.fromfile(fileName, dtype=np.uint8), cv.IMREAD_COLOR)
            imgCV = cv.imread(fileName)
        except:
            print("Open file fail!")
            return False

        LPRPic.setPath(fileName)
        self.imgOri = self.resizePicture(imgCV)
        if self.imgOri is None:
            print("Load picture fail!")
            return False
        # self.imgOri = ImageTk.PhotoImage(self.imgOri)
        self.labelPic.configure(image=self.imgOri, bg="pink")

        # ??????????????????
        cardColor = vehicleLicense.getColor(fileName)
        self.entryPlateColor.delete(0, END)  # ???????????????????????????
        self.entryPlateColor.insert(END, cardColor)

        # ??????????????????
        # ????????????????????????
        dist, car_plate = vehicleLicense.PlateRecognize(fileName, cardColor)
        # ???????????????????????????
        res = vehicleLicense.characterRecognize(car_plate)
        # ????????????????????????
        if len(res) == 0:
            self.entryPlateNum.delete(0, END)  # ???????????????????????????
            self.entryPlateNum.insert(END, "Identification failed")
        else:
            num = res[0][1][0]
            self.entryPlateNum.delete(0, END)  # ???????????????????????????
            self.entryPlateNum.insert(END, num)
        end = time.time()
        runTime = end - start
        print("???????????????", runTime)

    def __init__(self, *args, **kw):
        super().__init__()
        self.title("LPR Surface")
        self.geometry(str(self.tkWidth) + "x" + str(self.tkHeigth))
        self.resizable(0, 0)

        def labelInit():
            # Picture Label:
            self.labelPic = Label(self, text="Show Picture Area", font=("Arial", 24), bg="sky blue")
            self.labelPic.place(x=0, y=0, width=self.labelPicWidth, height=self.labelPicHeight)

            # Vehicle Plate Number Label:
            self.labelPlateNum = Label(self, text="Vehicle License Plate Number:", anchor=SW)
            self.labelPlateNum.place(x=0, y=self.labelPicHeight,
                                     width=self.textWidth * 20, height=self.textHeight)

            # Vehicle Colour Label:
            self.labelPlateCol = Label(self, text="Vehicle License Plate Color:", anchor=SW)
            self.labelPlateCol.place(x=0, y=self.labelPicHeight + self.textHeight * 2,
                                     width=self.textWidth * 20, height=self.textHeight)

        def buttonInit():
            # Picture Button
            self.buttonPic = Button(self, text="Load Picture", command=self.loadPicture)
            self.buttonPic.place(x=self.tkWidth - 3 * self.buttonWidth / 2,
                                 y=self.labelPicHeight + self.buttonHeight / 2,
                                 width=self.buttonWidth, height=self.buttonHeight)

        def entryInit():
            # Vehicle Plate Number Output
            self.entryPlateNum = Entry(self)
            # self.entryPlateNum = entryPlateNum
            self.entryPlateNum.place(x=self.textWidth, y=self.labelPicHeight + self.textHeight,
                                     width=self.textWidth * (42 - 1), height=self.textHeight)
            self.entryPlateNum.insert(END, "")

            # Vehicle Plate Color Output
            self.entryPlateColor = Entry(self)
            self.entryPlateColor.place(x=0, y=self.labelPicHeight + self.textHeight * 3,
                                       width=self.textWidth * (42 - 1), height=self.textHeight)
            self.entryPlateColor.insert(END, "")

        labelInit()
        buttonInit()
        entryInit()

        print("-------------init success-------------")
        self.mainloop()


if __name__ == '__main__':
    LS = LPRSurface()
    print("Finish")
