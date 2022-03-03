import pandas as pd
import numpy as np
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, precision_score, f1_score, roc_auc_score, accuracy_score

def process_data(train:str, test:str):
    #read from supabase
    train_data = pd.read_csv(train)
    print('original_training_dataset')
    train_data.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)
    print(train_data.shape)
    test_data = pd.read_csv(test)
    print('----------------------------')
    print('original_test_dataset')
    test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    print(test_data.shape)
    print(test_data.head())
    #data preprocessing begins
    # Sex Mapping 
    sex_mapping = {'male':0, 'female':1}
    train_data.Sex = train_data.Sex.map(sex_mapping)
    test_data.Sex = test_data.Sex.map(sex_mapping)
    
    # Map Embarked with numerical values - use in filling missing values of age using iterative imputer 
    embarked_mapping = {'C':0, 'Q':1, 'S':2}
    train_data.Embarked = train_data.Embarked.map(embarked_mapping)
    test_data.Embarked = test_data.Embarked.map(embarked_mapping)

    # Fill missing values of embarked and fare with median value
    train_data['Embarked'].fillna(value=train_data['Embarked'].median(), inplace=True)
    test_data['Embarked'].fillna(value=train_data['Embarked'].median(), inplace=True)
    train_data['Fare'].fillna(value=train_data['Fare'].median(), inplace=True)
    test_data['Fare'].fillna(value=train_data['Fare'].median(), inplace=True)

    # Create New Feature Family Size
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1 # 1 for childer with nanny
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

    # Converts Embarked into dummy
    train_data = pd.get_dummies(train_data)
    test_data = pd.get_dummies(test_data)

    #Handle missing values of Age using Iterative Impute
    useless_features = ['Survived', 'PassengerId']
    useful_features = [i for i in train_data.columns if i not in useless_features]

    # Fill missing value for Age using Iterative Imputer
    imputer = IterativeImputer(max_iter=25, random_state=42)

    train_data_imptr = imputer.fit_transform(train_data[useful_features])
    train_data_imtr = pd.DataFrame(train_data_imptr, columns = useful_features)
    train_data = train_data.drop(useful_features, axis=1)
    train_data = pd.concat([train_data, train_data_imtr], axis=1)

    test_data_imptr = imputer.transform(test_data[useful_features])
    test_data_imtr = pd.DataFrame(test_data_imptr, columns= useful_features)
    test_data = test_data.drop(useful_features, axis=1)
    test_data = pd.concat([test_data, test_data_imtr], axis=1)

    #Convert Features as per it's appropriate dtype
    train_data['Survived'] = train_data['Survived'].astype(int)
    train_data['Pclass'] = train_data['Pclass'].astype(int)
    train_data['SibSp'] = train_data['SibSp'].astype(int)
    train_data['Parch'] = train_data['Parch'].astype(int)
    train_data['Embarked'] = train_data['Embarked'].astype(int)
    train_data['FamilySize'] = train_data['FamilySize'].astype(int)

    test_data['PassengerId'] = test_data['PassengerId'].astype(int)
    test_data['Pclass'] = test_data['Pclass'].astype(int)
    test_data['SibSp'] = test_data['SibSp'].astype(int)
    test_data['Parch'] = test_data['Parch'].astype(int)
    test_data['Embarked'] = test_data['Embarked'].astype(int)
    test_data['FamilySize'] = test_data['FamilySize'].astype(int)

    print(train_data.shape, test_data.shape)

    #Split Train Dataset into Features & Target
    y = train_data['Survived']
    X = train_data.drop(['Survived'], axis=1)
    X_test = test_data.drop(['PassengerId'], axis=1)
    print('after splitting dataset into features and target')
    print(X.shape, y.shape, X_test.shape)

    #Train Test Split - For Validation
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=42, test_size=0.25)
    print('after train test split for model training')
    print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)

    #save dataset for model training
    np.save('X_train.npy', X_train)
    np.save('X_valid.npy', X_valid)
    np.save('Y_train.npy', Y_train)
    np.save('Y_valid.npy', Y_valid)
    np.save('X_test.npy', X_test)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata_url')
    parser.add_argument('--testdata_url')
    args = parser.parse_args()
    process_data(args.traindata_url, args.testdata_url)