import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={"account length":127,	"international plan":1,	"voice mail plan":1, "number vmail messages": 25,	"total day minutes":243.5,	"total day calls":110,	"total day charge":45.5,	"total eve minutes":195.5,	"total eve calls":100,	"total eve charge":16,	"total night minutes":250,	"total night calls":90,	"total night charge":9,	"total intl minutes":11, "total intl calls":5,	"total intl charge":3,	"customer service calls":3})

print(r.json())