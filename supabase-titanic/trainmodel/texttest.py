
# import urllib library
from urllib.request import urlopen
  
# import json
import json
# store the URL in url as 
# parameter for urlopen
url = "https://nqdmbmunxbbosfjjvhdj.supabase.in/storage/v1/object/public/demo/params.json"
  
# store the response of URL
response = urlopen(url)
  
# storing the JSON response 
# from url in data
params = json.loads(response.read())
  
# print the json response
print(params)
print(params["random_state"])