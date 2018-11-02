import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import json
from PIL import Image
from pixels_match import converter_pdf
import os
from os.path import join, dirname, realpath
import time
from pixels_match import converter_pdf,resizer_img,sub_image
import time
import cv2
import numpy as np
import math
import logging
from pixels_match import process_image

logging.basicConfig(level=logging.DEBUG)
import numpy as np

fn=""

def get():
    return fn

import sys


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def home_page():
    print("cd")
    return render_template('index1.html')

def html(content):  # Also allows you to set your own <head></head> etc
   return '<html><head><style> img {     border: 1px solid #ddd; /* Gray border */     border-radius: 4px;  /* Rounded border */     padding: 5px; /* Some padding */    width: 300px; /* Set a small width */ }/* Add a hover effect (blue shadow) */img:hover {    box-shadow: 0 0 2px 1px rgba(0, 140, 186, 0.5);}</style></head><body>' + content + '</body></html>'


def display(app):
    import os
    aa=os.listdir("C://Users//benna//OneDrive//Bureau//Drawapp//static//img//croped_img")
    
    body=""
    img_tag='<img src="'
    for name in aa:
        name=os.path.join("static/img/","croped_img/", name)
        tag=img_tag+name+'" alt="Girl in a jacket">'
        body+=tag
    return html(body)

def check_file(app,request):
    if request.method == 'POST':
#        if 'file' not in request.files:
#            print ('no file')
            
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_name = file.filename
            path=os.path.join(app.config['UPLOAD_FOLDER'],"check/")
            path_src=os.path.join(app.config['UPLOAD_FOLDER'],"check/", file_name)
            path_dst=os.path.join(app.config['UPLOAD_FOLDER'],"check_convert/")
            file.save(path_src)
            
            try:
                return process_image(path,file_name,path_dst)
            except:
                return "kkkkkk"
            
        
        

def upload_convert_pdf(app,request):
    print(request)
    if request.method == 'POST':
        if 'file' not in request.files:
            print ('no file')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_name = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))


            file_300=os.path.join(app.config['UPLOAD_FOLDER'],"img300/")
            file_200=os.path.join(app.config['UPLOAD_FOLDER'],"img200/")
            filename=file_name.replace(".pdf",".jpg")

            converter_pdf(app.config['UPLOAD_FOLDER'],file_name,file_300,filename,300)

            resizer_img(file_300, filename, file_200, filename, 7)


            return filename

    return "Can't post file"


def get_first_info(app,request,filename):
    if request.method == 'POST':
        logo_coordination = request.form['name_text']
        degree=request.form['rodio_stat']
        
        fn=logo_coordination
        out = logo_coordination
        o =json.loads(logo_coordination)

        print("=================")
        print(o["rotate"])
        print("===========")

    #    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], "img300\\", filename),o)
   #     print(o,file=sys.stdout)
    #    sub_image(os.path.join(app.config['UPLOAD_FOLDER'], "img300\\", filename),o,os.path.join(app.config['UPLOAD_FOLDER'], "croped_img\\", filename))
        
        img_1 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], "img200\\", filename))
        
        img2 = img_1.crop((o["x"] , o["y"] , o["x"]  + o["width"] , o["y"] + o["height"] ))
        
        img2.save(os.path.join(app.config['UPLOAD_FOLDER'], "croped_img_200\\", filename), "JPEG")

        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], "img300\\", filename))
        
        img2 = img.crop((o["x"] * 7, o["y"] * 7, o["x"] * 7 + o["width"] * 7, o["y"] * 7 + o["height"] * 7))
        
        print (degree)
        if int(degree)==90:
            img2=img2.rotate(90, expand=1)
            

        img2.save(os.path.join(app.config['UPLOAD_FOLDER'], "croped_img\\", filename), "JPEG")
        


        return render_template('crop_sec.html',msg=str(filename))
        return 'a string'

    return "Can't post file"


'''def get_secent_info(app,request):
    if request.method == 'POST':

        split_img = request.form['name_text']
        json_info['coo_img']=split_img
        o = json.loads(json_info['coo_img'])

        if file and allowed_file(file.filename):
      #      img=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'],"images200\\", filename))

       #     img=rescale(img,o["scalex"],o["scaley"])
       #     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],"images200\\", filename),img)

            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'],"images200\\", filename))


            img2 = img.crop((o["x"]*5, o["y"]*5, o["x"]*5+o["width"]*5, o["y"]*5+o["height"]*5))
         #   img2.show()

            img2.save(os.path.join(app.config['UPLOAD_FOLDER'],"rlogo\\"+ json_info['filename']), "JPEG")
          #  img2.save("C:/test"+ ".thumbnail", "JPEG")


            return render_template('secend_page.html',msg=str("rlogo\\"+json_info['filename']))

 #   return '''
 #   <!doctype html>

 #   '''

def crop(app,request,json_info):
    if request.method == 'POST':
        title = request.form['drawing_title']
        json_info['title'] = title

        revision=request.form['revision']
        json_info['revsion'] = revision

        price=request.form['drawing_number']
        json_info['price'] = price

        proj_num=request.form['project_number']
        json_info['project_number'] = price

        with open('{}/{}.json'.format(app.config['UPLOAD_FOLDER'],"rlogo\\"+json_info['filename'].split(".")[0]), 'w') as outfile:
            json.dump(json_info, outfile)


        json_info["filename"]=""
        json_info["coo_img"]=""
        json_info["title"]=""
        json_info["revsion"]=""
        json_info["price"]=""
        json_info["project_number"]=""


        print (json_info)

        return redirect("/")


def rescale(img,scal_x,scal_y):
    if (scal_x==1 and scal_y==1):
        return img
    elif (scal_x==-1 and scal_y==1):
        img=cv2.flip( img, 0 )
        return img
    elif (scal_x==1 and scal_y==-1):
        img=cv2.flip( img, 1 )
        return img
    elif (scal_x==-1 and scal_y==-1):
        img=cv2.flip( img, -1 )
        return img


def rotate(origin, point,angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

