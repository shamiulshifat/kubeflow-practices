import pandas as pd
import numpy as np
import argparse
import pickle
import time
#from mega import Mega
def deploy(X_test_data, model_file):
    start_time = time.time()
    X_test=np.load(X_test_data)
    print("-xtest-- %s seconds ---" % (time.time() - start_time))
    #features = ["Pclass", "SibSp", "Parch","Sex", "Age", "Fare"] # Important Features
    start_time1 = time.time()
    model = pickle.load(open(model_file, 'rb'))
    print("-modelload-- %s seconds ---" % (time.time() - start_time1))
    start_time2 = time.time()
    predictions = model.predict(X_test)
    print("-predict-- %s seconds ---" % (time.time() - start_time2))
    start_time3 = time.time()
    output = pd.DataFrame({'Survived': predictions})
    print("-predict-- %s seconds ---" % (time.time() - start_time3))
    start_time4 = time.time()
    output.to_csv('titanic_predictions.csv', index=False)
    print("-predict-- %s seconds ---" % (time.time() - start_time4))
    start_time5 = time.time()
    print("total survived:", output.Survived.value_counts())
    print("Your submission was successfully saved!")
    print("-submission-- %s seconds ---" % (time.time() - start_time5))
    #send to mega
    #start_time6 = time.time()
    #mega = Mega()
    #email='shifat@betterdata.ai'
    #password='shadia1afshan2raisha3'
    #print("-mega-- %s seconds ---" % (time.time() - start_time6))
    #start_time7 = time.time()
    #m = mega.login(email, password)
    #print("-login-- %s seconds ---" % (time.time() - start_time7))
    #start_time8 = time.time()
    #file_name='titanic_predictions.csv'
    #m.upload(file_name)
    #print("-upload-- %s seconds ---" % (time.time() - start_time8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    start_time9 = time.time()
    deploy(args.X_test, args.model)
    print("-deploy-- %s seconds ---" % (time.time() - start_time9))


