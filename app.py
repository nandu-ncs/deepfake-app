from flask import Flask
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.utils  import secure_filename
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():    
    return render_template('index.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('err404.html'), 404