#from xml.etree.ElementTree import tostring
from flask import Flask, request, render_template, jsonify
# import gridfs
from pymongo import MongoClient
#from werkzeug.utils import secure_filename
#import os
from model.yolo import yoloModel
from model.word2vec import Myword2vec
from model.lda import lda_model
#import json
#from markupsafe import escape

# client = MongoClient('0.0.0.0', 27017)
# mongodb 저장 경로
# db = client.insta
# collection = db['insta']

server = Flask(__name__)
server.config['JSON_AS_ASCII'] == False


@server.route('/')
def main():
    return render_template("index.html")


@server.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        upload_folder = 'static/uploads/'
        file.save(str(upload_folder+file.filename))
        # print('file.filename : ', file.filename) # party.jpg
        display = 'show'
        # print('file : ', file) # <FileStorage: 'party.jpg' ('image/jpeg')>
        unique_list_kr, tokenized_doc, tok_list = yoloModel(file.filename)
        keyword = Myword2vec(tok_list)
        lda_json = lda_model(keyword, tokenized_doc)
        return render_template("index.html", before_image=file.filename, display=display, lda_json=lda_json, keyword=keyword)


@server.route('/chart_word')
def chart_word():
    return render_template("chart.html")


