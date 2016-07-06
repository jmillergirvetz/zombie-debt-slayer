###Zombie Debt Slayer Flask Application###
import flask
from flask import Flask, render_template, request, jsonify
import logging
from logging import Formatter, FileHandler
import os
import pymongo
import json
import time
from datetime import datetime
import socket
import requests
import bokeh
import hashlib
import pandas as pd
import numpy as np


#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)


client = pymongo.MongoClient()
db = client['zombie-debt-slayer']
coll = db['debts']

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

def load_pickle(picklefile):
    with open(picklefile) as f:
        pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/debt-slayer', methods=['GET', 'POST'])
def debt_slayer():
    if request.method == 'GET':
        return render_template('debt-slayer.html')
    elif request.method == 'POST':
        # need to take list of inputs and make numpy array into x to then
        # transform into results
        # req_list = [str(request.form['my_debt']), str(request.form['my_debt'])]

        # apply cleaning to it
        zip_code = np.array[str(request.form['zip'])]
        # apply cleaning to it
        rf = load_pickle('../data/rf_model.pkl')
        x_transform = rf.transform(x)
        rf_pred = rf.predict(x_transform)

        # apply clearning if necessary
        narrative = [str(request.form['narr'])]
        # apply clearning if necessary
        model = load_pickle('../data/narrative_model.pkl')
        X = model.transform(text)
        predict = model.predict(X)
        return render_template('submissions.html', zip=zip_code, \
                                                   narr_predict=predict, text=text)

    return render_template('debt-slayer.html')

@app.route('/submissions')
def submissions():
    return render_template('submissions.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
