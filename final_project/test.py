import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

wine_instance = {
    'alcohol': 11.5,
    'sulphates': 0.74,
    'citric acid': 0.66,
    'volatile acidity': 0.53
}

data = {
  "wine": wine_instance
}

result = requests.post(url, json=data).json()
print(result)

