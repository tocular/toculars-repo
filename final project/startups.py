import flask
app = flask.Flask(__name__)

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import VotingRegressor

with open('ensemble_model.pkl', 'rb') as picklefile:
    ensemble_model = pickle.load(picklefile)

@app.route('/page')
def page():
   with open("startups.html", 'r') as viz_file:
     return viz_file.read()

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':
       inputs = flask.request.form
    
       country_hq = inputs['country_hq'][0]
       sector = inputs['sector'][0]
       continent = inputs['continent'][0]
       launch = inputs['launch'][0]
       woman_ceo = inputs['woman_ceo'][0]
       gender_mix = inputs['gender_mix'][0]
       round_type = inputs['round_type'][0]
       founders = inputs['founders'][0]
    
       item = np.array([])
       
       # adding value for launch
       item = np.append(item, [launch])
       
       # woman ceo
       if woman_ceo == 'Yes':
          item = np.append(item, [1])
       else:
          item = np.append(item, [0])
       
       # founders + hard-coded deal year 
       item = np.append(item, [founders, 0, 0, 1])
    
       # country hq
       if country_hq == 'Nigeria':
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       elif country_hq == 'Kenya':
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       elif country_hq == 'Egypt':
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       else:
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
     
       # sector
       if sector == 'Fintech':
          item = np.append(item, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
       elif sector == 'Logistics & Transport':
          item = np.append(item, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
       elif sector == 'Healthcare':
          item = np.append(item, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
       else:
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
       # ceo school
       if continent == 'Africa':
          item = np.append(item, [0, 0, 0, 0])
       elif continent == 'Europe':
          item = np.append(item, [0, 1, 0, 0])
       else:
          item = np.append(item, [0, 0, 0, 1])
            
      # gender mix
       if gender_mix == 'Male-only founding team':
          item = np.append(item, [0,1])
       elif gender_mix == 'Female-only founding team': 
          item = np.append(item, [0,0])
       else: 
          item = np.append(item, [1,0])
       
     # round type
       if round_type == 'Pre-Seed':
          item = np.append(item, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       elif round_type == 'Seed':
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
       elif round_type == 'Venture Round':
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
       elif round_type == 'Grant':
          item = np.append(item, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       else:
          item = np.append(item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
       item = item.reshape(1, -1)
       score = ensemble_model.predict(item)
       raw_score = (np.e)**score[0]
       raw_score = raw_score.round(2)
       return f"The predicted amount raised is {raw_score} million USD."


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'
    app.run(HOST, PORT, debug = True)