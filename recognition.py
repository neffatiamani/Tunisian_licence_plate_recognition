import numpy as np
import cv2
from PIL import Image , ImageDraw , ImageFont
import pytesseract
from pytesseract import Output
import pyocr
import argparse
from spellchecker import SpellChecker
import arabic_reshaper 
from bidi.algorithm import get_display
import time

start_time = time.perf_counter()


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
spell = SpellChecker()
#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True,
#                help = 'path to input image')
#args = ap.parse_args()
img =cv2.imread("result.jpg")
crop_img= cv2.imread('cropped.jpg')
# resize image
resized = cv2.resize(crop_img, (800, 170), interpolation = cv2.INTER_LINEAR_EXACT)
resized= cv2.copyMakeBorder(resized, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
print('Resized Dimensions : ',resized.shape)
#cv2.imshow("Resized image", resized)
cv2.imwrite("Resized.jpg", resized)
#cv2.waitKey(0)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # RGB to Gray scale conversion
blur = cv2.bilateralFilter(gray, 1 , 255, 255) 
blur = cv2.GaussianBlur(blur,(7,7),3)
blur = cv2.medianBlur(blur,5)
# perform otsu thresh (using binary inverse since opencv contours work better with white text)
thresh = cv2.threshold(blur, 200, 255,  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# median blurring should be done to remove noise
thresh = cv2.medianBlur(thresh,5)
thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
thresh = cv2.circle(thresh,(315,47),25,(255,255,255),-1) 
thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

# Morph open to remove noise and invert image 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,9))
dilation = cv2.dilate(thresh, kernel, iterations= 2)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN , kernel, iterations=1)
closing=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)


cv2.imshow('opening', opening)
cv2.imshow("dilation", dilation)
cv2.imshow('thresh', thresh)
cv2.imshow("blur",blur)
cv2.imshow('gray',gray)
cv2.imshow("closing",closing)
cv2.waitKey(3000)
cv2.destroyAllWindows()


#dst = cv2.Canny(closing, 50, 100)

dst = cv2.morphologyEx(closing, cv2.MORPH_RECT, np.zeros((3,3), np.uint8), iterations=1)
contours, heirarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[3])

rect=[]
ROI_number = 0

for i in sorted_contours:
    peri = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    if h >= 60  : # if height is enough
    # create rectangle for bounding
        rect = (x, y, w, h)
        cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,0,255),1)  
        ROI = closing[y:y+h, x:x+w] #crop each number individually  
        cv2.imwrite('character segmented n {}.jpg'.format(ROI_number), ROI)
        ROI_number += 1
        cv2.imwrite("contour.jpg", thresh_color) 

cv2.imshow('contr',thresh_color)
cv2.waitKey(200)

image=thresh.copy()
crop_img = image[0:300 , 280:460]
#crop_img = cv2.adaptiveThreshold(crop_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,1)
kernel1 = cv2.getStructuringElement(cv2.MORPH_DILATE, (4,7))
crop_img = cv2.dilate(crop_img, kernel1, iterations= 1)
cv2.imshow("cropped", crop_img)
txt1 = pytesseract.image_to_string(crop_img,lang='ara2' , config=' --psm 13 --oem 3 --dpi 300 ')
txt1 = spell.correction(txt1)
reshaped_text1 = arabic_reshaper.reshape(txt1)    # correct its shape'
bidi_text1 = get_display(reshaped_text1)
print( bidi_text1)
cv2.waitKey(200)
dst1 = cv2.morphologyEx(crop_img, cv2.MORPH_RECT, np.zeros((3,3), np.uint8), iterations=1)
contrs, heirarchy = cv2.findContours(dst1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sorted_contrs = sorted(contrs, key=lambda ctr: cv2.boundingRect(ctr)[3])
#rectt=[]
#ROI_number = 0
#for i in sorted_contrs:
#    x,y,w,h = cv2.boundingRect(i)
#    if w > 5 and h > 12:
#        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),1)  
#        ROI = crop_img[y:y+h, x:x+w]
#        cv2.imwrite('character n {}.jpg'.format(ROI_number), ROI)
#        ROI_number += 1
#        cv2.imshow("ggg",crop_img)

image1=closing.copy()
crop_img1 = image1[0:240 ,40:290]
#cv2.imshow("crop1", crop_img1)
#cv2.waitKey()
txt2 = pytesseract.image_to_string(crop_img1,lang='eng' , config=' --psm 13  --dpi 300 ')
print( txt2)

image3=closing.copy()
crop_img2 = image3[0:500 ,450:770]
#cv2.imshow("crop2", crop_img2)
#cv2.waitKey()
txt3 = pytesseract.image_to_string(crop_img2,lang='eng' , config=' --psm 11 --dpi 300')
print( txt3)
#a="تونس"
#a = spell.correction(a)
#a = arabic_reshaper.reshape(a)    # correct its shape
#a = get_display(a)
teext=str(txt2)+str(bidi_text1)+str(txt3)
#img= cv2.resize(img, (1300, 1000), interpolation = cv2.INTER_LINEAR_EXACT)
fontpath = "./arial.ttf" 
font = ImageFont.truetype(fontpath,50)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((x+55,h-y+50),teext,(0,0,250), font = font)
img = np.array(img_pil)
print("Recognized in : %s seconds ---" % (time.perf_counter() - start_time))
cv2.imshow('image finale', img) 
cv2.imwrite('image finale.jpg', img) 
cv2.waitKey()
