import pandas as pd
import numpy as np
import argparse
import pickle
from mega import Mega
def deploy(X_test_data, model_file):
    X_test=np.load(X_test_data)
    features = ["Pclass", "SibSp", "Parch","Sex", "Age", "Fare"] # Important Features
    model = pickle.load(open(model_file, "rb"))
    predictions = model.predict(X_test[features])
    output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': predictions})
    output.to_csv('titanic_predictions.csv', index=False)
    print("total survived:", output.Survived.value_counts())
    print("Your submission was successfully saved!")
    #send to mega
    mega = Mega()
    email='shifat@betterdata.ai'
    password='shadia1afshan2raisha3'
    m = mega.login(email, password)
    file_name='titanic_predictions.csv'
    m.upload(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    deploy(args.X_test, args.model)


