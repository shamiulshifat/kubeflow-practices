import pandas as pd
import numpy as np
from mega import Mega
import argparse

def process_data(link:str):
    #read from supabase
    df = pd.read_csv(link)
    print('original_dataset')
    print(df.to_string()) 

    #add a new column with random integers
    df['NewNumCol'] = np.random.choice([1, 9, 20], df.shape[0])
    print('modified_dataset')
    print(df.to_string()) 

    #send to mega
    mega = Mega()
    email='shifat@betterdata.ai'
    password='shadia1afshan2raisha3'
    m = mega.login(email, password)
    df.to_csv('salary_modified.csv')
    file_name='salary_modified.csv'
    
    m.upload(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_url')
    args = parser.parse_args()
    process_data(args.dataset_url)