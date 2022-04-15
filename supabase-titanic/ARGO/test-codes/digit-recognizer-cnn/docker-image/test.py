import pandas as pd
traindata_url="https://nqdmbmunxbbosfjjvhdj.supabase.in/storage/v1/object/public/demo/train.csv"
train = pd.read_csv(traindata_url)
print(train.shape)