# Predict Stock Prices using Deep Learning 
## Machine Learning Engineer Nanodegree 
## Capstone Project Proposal 


#### In this MLND capstone project, I will be attempting to predict future stock prices for selective stock symbols. I will take historical stock price data for specific Stock symbols from yahoo finance and use deep learning techniques to train a model. This model will then be used to predict prices for next couple of days. 
#### As a reference, I will be using Machine Learning techniques taught in Udacity’s Machine Learning NanoDegree and Machine Learning for Trading course. 
#### Python language and Keras6 library will be used for application development. 

### Problem Statement 
#### The problem to be solved in this project is to predict the future stock prices for next couple (5, 10, 20) of days.  
#### This problem will be structured as a supervised learning problem where the input dataset will be framed in such a way that it will have input and matching output values. A sliding window technique will be used to traverse input dataset and matching output. This will be considered a regression type of supervised learning as the output will be a continuous real value. 
#### The input dataset will be the historical daily prices(of the chosen Stock symbol) dataset downloaded from the Yahoo finance website8. Adjusted Close price for each day will be considered as an input. The input values indexed on date will be treated as a time series(sequence). This dataset will be split into training and test dataset. The training dataset will further be traversed using a sliding window technique. A sliding window with a fixed number of Adj Close values will be the input values and fixed number( depending upon the number of values to be predicted) of values immediately after the sliding window in sequence will be the matching output values. 
#### A model will be developed matching input window values to the output values using Deep Learning. After developing the model using training dataset, stock price will be predicted for the period starting where the training dataset period ends. The predicted values can be compared against the testing dataset values for evaluation. The closeness of predicted values to actual values can be measured using Kera’s evaluate method or using RMSE method. 
 
### Datasets and Inputs 
#### Data for the project is taken from Yahoo finance website8. Historical data of following stock tickers is taken – S&P500, AAPL, GOOG, MSFT, XOM, MYL, WMT, PFE, IBM. 
#### All available historical data for these stocks from yahoo finance website8 will be taken. Different stocks have different periods of data available. 
 
### Solution Statement 
#### In this project, attempt will be made to predict the prices using Deep Learning techniques of Recurrent Neural Networks. Recurrent Neural Network model LSTM (Long short term Memory) will be used to predict the stock prices. 
 
### Benchmark Model 
#### Multiple benchmark models will be used for comparison: 
#### 1) As we are using existing data and we have test data available for which actual values are already available. We can test our model generated predicted values with these actual values and can see how accurate our model is.  
#### 2) Online website - https://www.stock-forecasting.com/Content/Data/Test.aspx 
#### 3) Find\develop prediction model using KNN(K nearest neighbor) regression model and run the model on same data as to be used on Deep Learning model. KNN model will be used as a benchmark. 
#### 4) Use Logistic Regression prediction Model
 
### Evaluation Metrics 
#### Predicted prices will be compared to Actual historical prices. • Keras model.evaluate method will be used to find the training and test errors. • RMSE (Root mean Square Error) will be calculated for predicted and actual values.  
 
 
### Project Design 
#### Project will be implemented using programming language Python and neural network API Keras will be used for deep learning. 