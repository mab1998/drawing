import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
import time
from proccess import home_page,upload_convert_pdf,get_first_info,crop,get,check_file,display
from flask import Flask, flash, request, redirect, url_for, render_template
import logging
logging.basicConfig(level=logging.DEBUG)

json_info={
    "filename":"",
    "coo_img":"",
    "title":"",
    "revsion":"",
    "price":"",
    "project_number":""
}

UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/img/')


UPLOAD_FOLDER = '/tmp/flask-upload-test/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOADS_PATH


filename=None
@app.route('/check', methods=['GET', 'POST'])
def check():
    return check_file(app,request)

@app.route('/display')
def display_img():
    return display(app)

@app.route('/')
def first():
    return home_page()

@app.route('/second', methods=['GET', 'POST'])
def second():

    filename=upload_convert_pdf(app,request)
    json_info["filename"] = filename
    return render_template('crop_first_img.html', msg= filename)

@app.route('/third', methods=['GET', 'POST'])
def third():
    print(json_info["filename"])
    return get_first_info(app,request,json_info["filename"])


@app.route('/finish', methods=['GET', 'POST'])
def finish():
    print(get())
    json_info["coo_img"]=get()
    print(json_info["filename"])
    return crop(app,request,json_info)




if __name__ == "__main__":
	app.run(host="0.0.0.0", port=80,debug=True)
