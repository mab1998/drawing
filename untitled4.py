import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import json
from PIL import Image
from pixels_match import converter_pdf





UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER






import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
import time

json_info={
    "filename":"",
    "coo_img":"",
    "title":"",
    "revsion":"",
    "price":""
}

UPLOADS_PATH = join(dirname(realpath(__file__)), 'static\\img\\')


UPLOAD_FOLDER = '/tmp/flask-upload-test/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOADS_PATH


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



@app.route('/uplo', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print ('no file')
            return redirect(request.url)
        file = request.files['file']



        split_img = request.form['name_text']
        json_info['coo_img']=split_img
        print("===================")
        o = json.loads(json_info['coo_img'])



        print(json_info['coo_img'])
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print ('no filename')
            return redirect(request.url)
        if file and allowed_file(file.filename):
        #    print(file.filename)
         #   filename = secure_filename(file.filename)
            filename = file.filename
            print(filename)
            json_info['filename'] =".".join(filename.split(".")[:-2]) +"___"+ str(int(time.time()))+"."+filename.split(".")[-1]
            print(json_info["filename"])
            profilePic=json_info["filename"]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'],"images300\\", filename))
            img2 = img.crop((o["x"]*5, o["y"]*5, o["x"]*5+o["width"]*5, o["y"]*5+o["height"]*5))
         #   img2.show()

            img2.save(os.path.join(app.config['UPLOAD_FOLDER'],"rlogo\\"+ json_info['filename']), "JPEG")
          #  img2.save("C:/test"+ ".thumbnail", "JPEG")


            return render_template('secend_page.html',msg=str("rlogo\\"+json_info['filename']))

    return '''
    <!doctype html>

    '''

@app.route('/')
def studenta():
   return render_template('index1.html')

@app.route('/crope')
def studentza():
    return render_template('crop_first_img.html')

@app.route('/updf', methods=['GET', 'POST'])
def pdf():
    print(request.method)
    if request.method == 'POST':
        print (request.files)
        if 'file' not in request.files:
            print ('no file')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename.replace(".pdf",".jpg"))
            print(app.config['UPLOAD_FOLDER'])
            converter_pdf(app.config['UPLOAD_FOLDER'],filename,app.config['UPLOAD_FOLDER'],filename.replace(".pdf",".jpg"),50)



            print("file has been uploaded")

            print("eee")
            return render_template('crop_first_img.html',msg=str(filename.replace(".pdf",".jpg")))
            return 'a string'

    return "Can't post file"


@app.route('/updfc', methods=['GET', 'POST'])
def pdfs():
    print(request.method)
    if request.method == 'POST':
        split_img = request.form['name_text']

        return render_template('crop_sec.html',msg=str(filename.replace(".pdf",".jpg")))
        return 'a string'

    return "Can't post file"







@app.route('/finsh', methods=['GET', 'POST'])
def json_file():
    if request.method == 'POST':
        title = request.form['title']
        json_info['title'] = title

        revision=request.form['revision']
        json_info['revsion'] = revision

        price=request.form['price']
        json_info['price'] = price

        with open('{}/{}.json'.format(UPLOADS_PATH,"rlogo\\"+json_info['filename'].split(".")[0]), 'w') as outfile:
            json.dump(json_info, outfile)


        json_info["filename"]=""
        json_info["coo_img"]=""
        json_info["title"]=""
        json_info["revsion"]=""
        json_info["price"]=""


        print (json_info)

        return redirect("/")

@app.route('/image')






@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)







if __name__ == "__main__":
    app.run(debug=True)