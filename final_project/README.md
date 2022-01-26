# Red Wine Quality Prediction

The goal of this project is to determine which features of red wine variants of the Portuguese "Vinho Verde" wines contribute to the quality of a good wine and apply a model to predict the quality of a Portuguese "Vinho Verde" wine.

## Dataset

More information and related data can be found here: https://www.kaggle.com/ucim/red-wine-quality-cortez-et-al-2009 .

## Data Dictionary

**fixed acidity**
most acids involved with wine or fixed or nonvolatile (do not evaporate readily).

**volatile acidity**
the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste.

**citric acid**
found in small quantities, citric acid can add 'freshness' and flavor to wines.

**residual sugar**
the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet.

**chlorides**
the amount of salt in the wine.

**free sulfur dioxide**
the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine.

**total sulfur dioxide**
amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine.

**density**
the density of wine is close to that of water depending on the percent alcohol and sugar content.

**pH**
describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale.

**sulphates**
a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant.

**alcohol**
the percent alcohol content of the wine.

**quality**
output variable (based on sensory data, score between 0 and 10).

## How to Run Application

- Build and run the application using the following commands:

```
docker build -t red-wine-predictions .

docker run -it --rm -p 8080:8080 red-wine-predictions 
```

Then, Use test.py as input to the service

- Run locally
```
 python predict_app.py
 curl http://localhost:9696/predict
```

- Web Service
```
python predict_wine.py
Jupyter Statements
```
import requests
url = 'http://localhost:9696/predict'
wine = {
    "alcohol": 20.5,
    "sulphates": 0.74,
    "citric acid": 0.66,
    "volatile acidity": 0.04
}
requests.post(url, json=wine).json()
```
```
