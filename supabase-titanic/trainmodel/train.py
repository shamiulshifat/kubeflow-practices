import pandas as pd
import numpy as np
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
#import joblib
#from mega import Mega
from urllib.request import urlopen
import json
import pickle

def train(param_data, X_train_data, Y_train_data, X_valid_data, Y_valid_data):
    #load data
    X_train=np.load(X_train_data)
    Y_train=np.load(Y_train_data)
    X_valid=np.load(X_valid_data)
    Y_valid=np.load(Y_valid_data)
    #load parameter from txt
    # store the response of URL
    response = urlopen(param_data)
    # storing the JSON response 
    # from url in data
    params = json.loads(response.read())
    
    # print the json response
    #print(params)
    #print(params['random_state'])
    #end of params reading
    #train model
    features = ["Pclass", "SibSp", "Parch","Sex", "Age", "Fare"] # Important Features
    # Randomly Tune the paramters
    print('--------------')
    print(X_train)
    RFC = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=params["random_state"], max_depth=params["max_depth"])
    RFC.fit(X_train, Y_train)
    RFC_Predict = RFC.predict(X_valid)
    RFC_Accuracy = accuracy_score(Y_valid, RFC_Predict)
    print("Accuracy: " + str(RFC_Accuracy))
    RFC_AUC = roc_auc_score(Y_valid, RFC_Predict) 
    print("AUC: " + str(RFC_AUC))
    print("Precision Score :" + str(precision_score(Y_valid, RFC_Predict)))
    print("Recall Score :" + str(recall_score(Y_valid, RFC_Predict)))
    print("F1 Score :" + str(f1_score(Y_valid, RFC_Predict)))
    print(classification_report(Y_valid, RFC_Predict))
    
    #save model to cloud
    pickle.dump(RFC, open('model.pkl', 'wb'))
    #send to mega
    #mega = Mega()
    #email='shifat@betterdata.ai'
    #password='shadia1afshan2raisha3'
    #m = mega.login(email, password)
    #file_name='model.pkl'
    #m.upload(file_name)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param')
    parser.add_argument('--X_train')
    parser.add_argument('--Y_train')
    parser.add_argument('--X_valid')
    parser.add_argument('--Y_valid')
    args = parser.parse_args()
    train(args.param, args.X_train, args.Y_train, args.X_valid, args.Y_valid)