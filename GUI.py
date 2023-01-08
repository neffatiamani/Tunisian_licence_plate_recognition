import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from tkinter import PhotoImage
import numpy as np
import cv2
import pytesseract 
import os
import arabic_reshaper
from bidi.algorithm import get_display

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


window=tk.Tk()
window.geometry('1200x800')
window.title(' Automated Licence Plate Recognition ')
window.iconphoto(True, PhotoImage(file="logo.png"))
img = ImageTk.PhotoImage(Image.open("logo.png"))

window.configure(background='#97a4d9')
label=Label(window,background='#a5b0db', font=('Arial',50,'bold'))
# label.grid(row=0,column=1)
sign_image = Label(window,bd=10)
plate_image=Label(window,bd=10)

def classify(file_path):
    
    img=cv2.imread(file_path)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 15, 15)
    blur = cv2.medianBlur(blur,5)
    blur = cv2.GaussianBlur(blur,(5,5),0)
#thresh = cv2.adaptiveThreshold(blur,255,0,1,7,1)
    thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)



# Morph open to remove noise and invert image
#kernel = np.ones((4,6), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,9))
    dilation = cv2.dilate(thresh, kernel, iterations= 1 )

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN , kernel, iterations=1)
    closing=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    uploaded=Image.open("cropped.jpg")
    im=ImageTk.PhotoImage(uploaded)
    plate_image.configure(image=im)
    plate_image.image=im
    plate_image.pack()
    plate_image.place(x=500,y=500)
    
    txt=pytesseract.image_to_string(uploaded,lang='ara2' , config=' --psm 13 --oem 3 --dpi 300  ')
    reshaped_text = arabic_reshaper.reshape(txt)    # correct its shape
    bidi_text = get_display(reshaped_text)
    label.configure(foreground='#011638', text=bidi_text)
def show_classify_button(file_path):
    classify_b=Button(window,text="Classify Image",command=lambda:classify(file_path),padx=20,pady=7)
    classify_b.configure(background='#2f428d', foreground='white',font=('Elephant',18,'bold'))
    classify_b.place(x=70,y=500)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((window.winfo_width()/2.8),(window.winfo_height()/2.8)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass




upload=Button(window,text="Upload an Image",command=upload_image,padx=20,pady=7)
upload.configure(background='#2f428d', foreground='white',font=('Elephant',18,'bold'))
upload.pack()
upload.place(x=70,y=300)
sign_image.pack()
sign_image.place(x=500,y=200)
label.pack()
label.place(x=500,y=600)
heading = Label(window,image=img)
heading.configure(background='#2f428d',foreground='#bec7e9')
heading.pack()


window.mainloop()