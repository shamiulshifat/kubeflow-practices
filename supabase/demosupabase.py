import pandas as pd
import numpy as np
from mega import Mega

#read from supabase
df = pd.read_csv('https://nqdmbmunxbbosfjjvhdj.supabase.in/storage/v1/object/public/demo/Salary_Data.csv')

print(df.to_string()) 

#add a new column with random integers
df['NewNumCol'] = np.random.choice([1, 9, 20], df.shape[0])
print(df.to_string()) 

#send to mega
mega = Mega()
email='shifat@betterdata.ai'
password='shadia1afshan2raisha3'
m = mega.login(email, password)
df.to_csv('salary_modified.csv')
file_name='salary_modified.csv'

m.upload(file_name)





