# Home Price Prediction Project

This project is about predicting residential home prices in Durham, NC.

## Dataset

The dataset is data from the local multiple listing service for homes sold in Durham, NC from January 1, 2021 thru October 31, 2021.

## How to Run Application

Clone the directory into your work space.  UPdate the home dictionary in the predict-test.py file and issue

```
python predict-test.py
```
           or
           
Build and run the application using the following commands:

```
docker build -t midterm-project .

docker run -it --rm --p 9696:9696 midterm-project
```