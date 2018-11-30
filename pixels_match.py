# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 18:27:01 2018

@author: ambs
"""
import matplotlib.pyplot as plt


# import the necessary packages
import numpy as np
import imutils
import math
#import glob
import cv2
import os
from shutil import copyfile
from wand.image import Image as Img
import os
import cv2
import numpy as np
import os
from scipy.misc import imread, imshow, imsave
from scipy.ndimage import rotate
import json
import pytesseract
from PIL import Image

def sub_image(image, o,save_path):
    image = cv2.imread(image)

    x_center=(int(o["x"]* 7)+int(o["width"]* 7))/2
    y_center = (int(o["y"]* 7) + int(o["height"]* 7)) / 2
    center=(x_center,y_center)
    theta=int(o["rotate"])
    width=int(o["width"]* 7)
    height=int(o["height"]* 7)

    if 45 < theta <= 90:
        theta = theta - 90
        width, height = height, width

    theta *= math.pi / 180 # convert to rad
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])

    img=cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
 #   img.save(save_path)
    cv2.imwrite(save_path,img)
    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    #vc = cv2.VideoCapture(0)
def rotate_(path, angle):
    img = imread(path)

    rotate_img = rotate(img, -90)
    imsave(path, rotate_img)

def resizer_img(path_source,filname_source,path_destination,filname_destination ,scale):
        filname=path_source+filname_source
        print(filname)
        img_rgb = cv2.imread(filname)
        height, width =img_rgb.shape[:2]
        h_scal=int(height/scale)
        w_scal=int(width/scale)
        vvv_thumbnail = cv2.resize(img_rgb, (w_scal, h_scal), interpolation = cv2.INTER_AREA)
        filenamed=path_destination+filname_destination
        cv2.imwrite(filenamed, vvv_thumbnail)
        return filname_destination

def converter_pdf(path_source,pdffilename,path_destination,imgfilename,dpi):

    if "pdf" in pdffilename:
        filename=path_source+pdffilename
        image_name=path_destination+imgfilename

        with Img(filename=filename, resolution=dpi) as img:
            img.compression_quality = 99
           
            img.save(filename=image_name)
    return image_name

def found_rectlogo(path_main,image,file_name,alpha):
#    path="./logo-origin/"
#    path="./static/img/croped_img/"
#     path="./static/img/croped_img_200/"
    path=os.path.join(path_main,"croped_img_200/")
    #cv2.imshow("Template", template)
    count=0
    # loop over the images to find the template in
    found = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    rec_number=0
    listdir=os.listdir(path)
    print(listdir)
#    bar = Bar('Processing', max=len(listdir))
    for name in listdir:
    	template_path=path+name
#    	print(template_path)    
#    	template_source = cv2.imread(template_path)
    	im1=Image.open(template_path)
    	im2=im1.rotate(90, expand=1)

#    	template_source = cv2.imread(template_path)
    	template = np.array(im1) 
    	template90= np.array(im2) 
    	template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#    	template_source = Image.open(template_path)
#    	print(template.shape,'---',template_path)        
#    	template = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)
    	template = cv2.Canny(template_gray, 50, 200)
#    	imgplot = plt.imshow(template)

    	(tH, tW) = template.shape[:2]
        
    	(h, w) = (tH, tW)
    	center = (w / 2, h / 2)
 
#    	center = (w / 2, h / 2)
    	angle90 =270
    	scale = 1.0                
    	template_gray90 = cv2.cvtColor(template90, cv2.COLOR_BGR2GRAY)

#    	M = cv2.getRotationMatrix2D(center, angle90, scale)
#    	template90 = cv2.warpAffine(template_gray, M, (h, w))   
#    	img2=img2.rotate(90, expand=1)
    	(tH90, tW90) = template90.shape[:2]          

    	template90 = cv2.Canny(template_gray90, 50, 200)
#    	imgplot = plt.imshow(template90)
        
        
        
    	for scale in np.linspace(0.2, 1.0,20)[::-1]:
    		print('scale  ',scale)
    		# resize the image according to the scale, and keep track
    		# of the ratio of the resizing.
    		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))

    		r = gray.shape[1] / float(resized.shape[1])
#    		bar1.next()
     
    		# if the resized image is smaller than the template, then break
    		# from the loop
    		if resized.shape[0] < tH or resized.shape[1] < tW:
    			pass
    		else:
    			rotate=0           
        

        		# detect edges in the resized, grayscale image and apply template
        		# matching to find the template in the image
    			edged = cv2.Canny(resized, 50, 200)
    			result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 
    			if found is None or maxVal > found[0]:
    			    			found = (maxVal, maxLoc, r,tH, tW,name,rotate)
                
################################################################################
#
################################################################################
              
#    		for scale in np.linspace(0.2, 1.0,20)[::-1]:
                
            		# resize the image according to the scale, and keep track
            		# of the ratio of the resizing
         
#    		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
#    		r = gray.shape[1] / float(resized.shape[1])
#    		bar1.next()
     
    		# if the resized image is smaller than the template, then break
    		# from the loop
    		if resized.shape[0] < tH90 or resized.shape[1] < tW90:
    			continue
    		# detect edges in the resized, grayscale image and apply template
    		# matching to find the template in the image
#    		edged = cv2.Canny(resized, 50, 200)
    		result = cv2.matchTemplate(edged, template90, cv2.TM_CCOEFF)
    		(_, maxVal90, _, maxLoc90) = cv2.minMaxLoc(result)
#    		print(maxVal)            
     
    		if maxVal > found[0]:
    			rotate=1                  
    			found = (maxVal90, maxLoc90, r,tH90, tW90,name,rotate)
#    			print(maxVal)       
#################################################################################
#
################################################################################    
#    	if found is None or maxVal > found[0]:
#    			found = (maxVal, maxLoc, r,tH, tW,name,rotate) 
    	# unpack the bookkeeping varaible and compute the (x, y) coordinates
    	# of the bounding box based on the resized ratio
    (_, maxLoc, r,tH, tW,name,rotate) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    print('max==',maxVal) 
    	# draw a bounding box around the detected result and display the image
    rect=image[startY:endY,startX:endX]
#    imgplot = plt.imshow(rect)
#    cv2.imwrite('./rect0/{}'.format(name),rect)

    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    rect=image[startY:endY,startX:endX]
    path0=os.path.join(path_main,"check_convert_300/")
    rect_temp=cv2.imread(path0+file_name)
    x0=startX*alpha
    x1=endX*alpha
    y0=startY*alpha
    y1=endY*alpha
    rect0=rect_temp[y0:y1,x0:x1]
    if rotate==1:
        im_pil0 = Image.fromarray(rect0)
        im_pil0=im_pil0.rotate(90, expand=1)
        rect0 = np.asarray(im_pil0)


# For reversing the operation:

    count=count+1    
#    cv2.waitKey(0)
#    return  image,rect,rect0,name
    return  rect0,name

def resize_folder(path_source,path_destination,scale):
    listdir=os.listdir(path_source)
#    bar = Bar('Processing', max=len(listdir))
    for name in listdir:
        if 'jpg' in name:
            resizer_img(path_source,name,path_destination,name,scale)

def get_rect(data_):
#    print(data_)
    data_=json.loads(data_)
    x_start=int(data_['x'])
    y_start=int(data_['y'])
    x_end=x_start+int(data_['width'])
    y_end=y_start+int(data_['height'])
    rotate=int(data_['rotate'])
    return x_start,y_start,x_end,y_end,rotate    
def ocr(image):
#        path_img=path+'wline/'+'_'+name
#        print('#######################################################')
#        print(path_img)
        
        
        
        
#        image = cv2.imread(path_img)
#        name_txt=path_img.replace('.png','.txt')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         
        # check to see if we should apply thresholding to preprocess the
        # image
        #if args["preprocess"] == "thresh":
#        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
         
        # make a check to see if median blurring should be done to remove
        # noise
        #elif args["preprocess"] == "blur":
#        gray = cv2.medianBlur(gray,3 )
         
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)
        
#        im = Image.open(path_img) # the second one 
#        im = im.filter(ImageFilter.MedianFilter())
#        enhancer = ImageEnhance.Contrast(im)
#        im = enhancer.enhance(2)
#        im = im.convert('1')
#        im.save('temp2.jpg')
        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        
        tessdata_dir_config = '--oem 3 --psm 11  --tessdata-dir "C://Program Files (x86)//Tesseract-OCR//tessdata//"'
        
        text = pytesseract.image_to_string(Image.open('./'+str(filename)),lang='eng', config=tessdata_dir_config)
        return text

def get_data(path_main,image,name0):
        profile_ocr={}
        profile_ocr['text']=[]
        profile_ocr['text'].append({}) 
#        name="1409-004-Key-Elevation-Bank-.json"

        # path_json="./static/img/rlogo/"
        path_json = os.path.join(path_main, "rlogo/")
        # path_logo="./static/img/croped_img_300/"
        path_logo = os.path.join(path_main, "croped_img_300/")
        # path_img="./static/img/croped_img_300/"
        path_img = os.path.join(path_main, "croped_img_300/")

        
        file_json=name0.replace('jpg','json')
        file_json=path_json+file_json
        file_logo=file_json.replace('json','jpg').replace('rlogo','croped_img')
#        file_logo=path_logo+file_logo
        print(name0)
        logo= cv2.imread(file_logo)
#        imgplot = plt.imshow(logo)
#        plt.show()
        w0,h0=logo.shape[:2]        
        w1,h1=image.shape[:2]
        scale_w=(w1/w0)
        scale_h=(h1/h0)
        
        with open(file_json) as datafile:
                data = json.load(datafile)
        
#        img_rgb = cv2.imread(fileImg)
                        #
        #print(data)
        title = data['title']
        print('title   :',title)
        x_start,y_start,x_end,y_end,rotate=get_rect(title)        
        img_title=cv2.rectangle(image, (int(x_start*scale_w), int(y_start*scale_h)), (int(x_end*scale_w), int(y_end*scale_h)), (0, 255, 0), 7)
        img_title=image[int(y_start*scale_h):int(y_end*scale_h),int(x_start*scale_w):int(x_end*scale_w)]
#        imgplot = plt.imshow(img_title)
#        plt.show()
        title_text=ocr(img_title)

        drawingN=data['project_number']
        x_start,y_start,x_end,y_end,rotate=get_rect(drawingN)
#        img_drawingN=cv2.rectangle(img_rgb, (x_start*scale_w, y_start*scale_h), (x_end*scale_w, y_end*scale_h), (0, 0, 255), 2)
        img_drawingN=image[int(y_start*scale_h):int(y_end*scale_h),int(x_start*scale_w):int(x_end*scale_w)]
        drawingN_text=ocr(img_drawingN)
        

        
        revsion=data['revsion']
        x_start,y_start,x_end,y_end,rotate=get_rect(revsion)        
#        img_revsion=cv2.rectangle(img_rgb, (x_start*scale_w, y_start*scale_h), (x_end*scale_w, y_end*scale_h), (0, 0, 255), 2)
        img_revsion=image[int(y_start*scale_h):int(y_end*scale_h),int(x_start*scale_w):int(x_end*scale_w)] 
        revsion_text=ocr(img_revsion)
        
        profile_ocr['text'][0]['Title']=title_text
        profile_ocr['text'][0]['drawingN']=drawingN_text
        profile_ocr['text'][0]['revsion']=revsion_text
        
        total_data=(json.dumps(profile_ocr,indent=4,ensure_ascii=False))

        return total_data

def process_image(path_source,pdffilename,path_destination0):
    path_source = './static/img/check/'
    # path_destination0 = './static/img/check_convert_300/'
    path_destination1 = './static/img/check_convert_200/'
    # pdffilename = '00.pdf'
    imgfilename = pdffilename.replace('pdf', 'jpg')
    alpha = 7
    dpi = 300
    # data = process_image(path_destination1, imgfilename, alpha)
    file_name_img=path_destination1+imgfilename
    img_rgb=cv2.imread(file_name_img)
    image=img_rgb
    file_name=	imgfilename
    image,rect,rect0,name0=found_rectlogo(img_rgb,imgfilename,alpha)
    print(name0)
#    image=found_rectlogo(img_rgb,name)
    
    json_data=get_data(rect0,name0)
    print(json_data)
    return json_data


# #
# path_source='./static/img/check/'
# path_destination0='./static/img/check_convert_300/'
# path_destination1='./static/img/check_convert_200/'
#
# #    path_source='C:/Users/benna/OneDrive/Bureau/Drawapp/static/img/check/'
#
# pdffilename='00.pdf'
# #    path_destination='C:/Users/benna/OneDrive/Bureau/Drawapp/static/img/check_convert/'
# imgfilename=pdffilename.replace('pdf','jpg')
# alpha=7
# dpi=300
# filname=converter_pdf(path_source,pdffilename,path_destination0,imgfilename,dpi)
#
# resizer_img(path_destination0,imgfilename,path_destination1,imgfilename ,alpha)
# data=process_image(path_destination1,imgfilename,alpha)
# print(data)