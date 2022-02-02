#import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle
#kfp imports
from typing import NamedTuple
import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op


######################dataset

#path = "data/"





#load data
@func_to_container_op
def load_data(traindata:str, testdata:str, train_out:OutputPath(str), test_out:OutputPath(str)):
    train_df=pd.read_csv(traindata)
    test_df=pd.read_csv(testdata)
    train_df. to_pickle("train_out.pkl")
    test_df. to_pickle("test_out.pkl")




################data processing
@func_to_container_op
def process_data(train_data:InputPath(str), test_data:InputPath(str), traindf_proc:OutputPath(str), testdf_proc:OutputPath(str)):
    train_df=pd. read_pickle(train_data)
    test_df= pd. read_pickle(test_data)
    #SibSp and Parch
    #SibSp and Parch
    data = [train_df, test_df]
    for dataset in data:
        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
        dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
        dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
        dataset['not_alone'] = dataset['not_alone'].astype(int)
    train_df['not_alone'].value_counts()
    # Survival with respect to the number of relatives in the ship
    axes = sns.catplot('relatives','Survived', kind='point',
                        data=train_df, aspect = 2.5, )  

    # This does not contribute to a person survival probability
    train_df = train_df.drop(['PassengerId'], axis=1)  
    #Missing data: Cabin
    #Create a new Deck feature
    import re
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data = [train_df, test_df]
    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)
    
    # we can now drop the cabin feature
    train_df = train_df.drop(['Cabin'], axis=1)
    test_df = test_df.drop(['Cabin'], axis=1)

    #Missing data: Age
    #Fill missing data from age feature with a random sampling from the distribution of the existing values.
    data = [train_df, test_df]

    for dataset in data:
        mean = train_df["Age"].mean()
        std = test_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean - std, mean + std, size = is_null)
        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)
    
    train_df["Age"].isnull().sum()

    #Missing data: Embarked
    train_df['Embarked'].describe()
    # fill with most common value
    common_value = 'S'
    data = [train_df, test_df]

    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    
    train_df. to_pickle("traindf_proc.pkl")
    test_df. to_pickle("testdf_proc.pkl")



@func_to_container_op
def feature_engg(train_data:InputPath(str), test_data:InputPath(str), traindf_proc:OutputPath(str)):
    train_df=pd. read_pickle(train_data)
    test_df= pd. read_pickle(test_data)
    
    #convert features
    data = [train_df, test_df]
    #######################feature engineering
    for dataset in data:
        dataset['Fare'] = dataset['Fare'].fillna(0)
        dataset['Fare'] = dataset['Fare'].astype(int)

        
    #Titles features
    data = [train_df, test_df]
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                                'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)

            
    train_df = train_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)

    #Sex into numeric
    genders = {"male": 0, "female": 1}
    data = [train_df, test_df]
        
    for dataset in data:
        dataset['Sex'] = dataset['Sex'].map(genders)

    
    #Drop Ticket feature
    train_df = train_df.drop(['Ticket'], axis=1)
    test_df = test_df.drop(['Ticket'], axis=1)

    #Embarked into numeric
    ports = {"S": 0, "C": 1, "Q": 2}
    data = [train_df, test_df]

    for dataset in data:
      dataset['Embarked'] = dataset['Embarked'].map(ports)
    
    #Age into categories

    data = [train_df, test_df]
    for dataset in data:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

        
    # let's see how it's distributed train_df['Age'].value_counts()
    #Fare into categories
    data = [train_df, test_df]

    for dataset in data:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
        dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)

        
    #New Features
    #Age times Class
    data = [train_df, test_df]
    for dataset in data:
        dataset['Age_Class']= dataset['Age']* dataset['Pclass']

    #Fare per person
    for dataset in data:
        dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
        dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    # Let's take a last look at the training set, before we start training the models.
    #train_df.head(10)
    train_df. to_pickle("traindf_proc.pkl")


@func_to_container_op
def mlmodel_random(traindata:InputPath(str)):
    train_df=pd. read_pickle(traindata)
    #####################   ML Model  
    PREDICTION_LABEL = 'Survived'
    train_labels = train_df[PREDICTION_LABEL]
    train_df = train_df.drop(PREDICTION_LABEL, axis=1)
    #Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(train_df, train_labels)
    acc_random_forest = round(random_forest.score(train_df, train_labels) * 100, 2)
    return acc_random_forest


@func_to_container_op
def mlmodel_reg(traindata:InputPath(str)):
    train_df=pd. read_pickle(traindata)
    #####################   ML Model  
    PREDICTION_LABEL = 'Survived'
    train_labels = train_df[PREDICTION_LABEL]
    train_df = train_df.drop(PREDICTION_LABEL, axis=1)
    #Logistic Regression

    logreg = LogisticRegression(solver='lbfgs', max_iter=110)
    logreg.fit(train_df, train_labels)
    acc_log = round(logreg.score(train_df, train_labels) * 100, 2)
    return acc_log

    
@func_to_container_op
def mlmodel_bayes(traindata:InputPath(str)):
    train_df=pd. read_pickle(traindata)
    #####################   ML Model  
    PREDICTION_LABEL = 'Survived'
    train_labels = train_df[PREDICTION_LABEL]
    train_df = train_df.drop(PREDICTION_LABEL, axis=1)
    #Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(train_df, train_labels)
    acc_gaussian = round(gaussian.score(train_df, train_labels) * 100, 2)
    return acc_gaussian

@func_to_container_op
def mlmodel_svc(traindata:InputPath(str)):
    train_df=pd. read_pickle(traindata)
    #####################   ML Model  
    PREDICTION_LABEL = 'Survived'
    train_labels = train_df[PREDICTION_LABEL]
    train_df = train_df.drop(PREDICTION_LABEL, axis=1)
    #SVM
    linear_svc = SVC(gamma='auto')
    linear_svc.fit(train_df, train_labels)
    acc_linear_svc = round(linear_svc.score(train_df, train_labels) * 100, 2)
    return acc_linear_svc


@func_to_container_op
def print_results(data:InputPath(str)):

    print(pd. read_pickle(data))





#build pipeline
@dsl.pipeline(
   name='titanic pipeline',
   description='A demo pipeline for handling titanic.'
)
def make_pipeline():
    test_df ="https://raw.githubusercontent.com/kubeflow-kale/kale/master/examples/titanic-ml-dataset/data/test.csv"
    train_df ="https://raw.githubusercontent.com/kubeflow-kale/kale/master/examples/titanic-ml-dataset/data/train.csv"
    load_dataset=load_data(train_df, test_df)
    processed_data=process_data(load_dataset.outputs['train_out'], load_dataset.outputs['test_out'])
    model_random=mlmodel_random(processed_data.output)
    model_reg=mlmodel_reg(processed_data.output)
    model_bayes=mlmodel_bayes(processed_data.output)
    model_svc=mlmodel_svc(processed_data.output)
    results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'logistic Regression',
              'Random Forest', 'Naive Bayes'],
    'Score': [model_svc.output, model_reg.output,
              model_random.output, model_bayes.output]})
    result_df = results.sort_values(by='Score', ascending=False)
    
    result_df = result_df.set_index('Score')
    result_df.to_pickle('results.pkl')
    accuracy=print_results('results.pkl')


    
    


##compile to yaml file
# Compile the pipeline
pipeline_func = make_pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline_titanic.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)


