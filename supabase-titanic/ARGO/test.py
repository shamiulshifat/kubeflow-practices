# we test event based workflow execution


import requests

url = 'http://localhost:12000/deploy'
myobj = {"message":"hello betterdata"}

x = requests.post(url, data = myobj)

print(x.text)