# importing required libraries
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from app_helper import *


app = Flask(__name__)


@app.route("/")
def index():
    '''
    Function to render index.html
    '''
    return render_template("index.html")


@app.route('/uploader', methods=['POST'])
def upload_file():
    '''
    Function to send data to server (POST)
    '''
    if request.method == 'POST':
        f = request.files['file']

        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # get the prediction
        predicted_class_label = get_classes(file_path)
        predicted_class_label = 'Predicted Class: ' + predicted_class_label

    return render_template("upload.html", predictions=predicted_class_label,
                           display_image=secure_filename(f.filename))


if __name__ == "__main__":
    app.run()
