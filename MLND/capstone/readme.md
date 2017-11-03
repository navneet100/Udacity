<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Machine​ ​Learning​ ​Engineer​ ​Nanodegree
## Capstone​ ​Project
=======
# Machine​ ​Learning​ ​Engineer​ ​Nanodegree 
## Capstone​ ​Project 

## Predict​ ​Stock​ ​Prices​ ​using​ ​Deep​ ​Learning 

### Navneet​ ​Latawa 
### 28​ ​Sep​ ​2017 
 
## Definition 

#### In this MLND capstone project, I will be attempting to predict future stock prices for selective stock symbols. I will take historical stock price data for specific Stock symbols from yahoo finance and use deep learning techniques to train a model. This model will then be used to predict prices for next couple of​ ​days. As a reference, I will be using Machine Learning techniques taught in Udacity’s Machine Learning NanoDegree​ ​and​ ​Machine​ ​Learning​ ​for​ ​Trading​ ​course. Python​ ​language​ ​and​ ​Keras​​ library​ ​will​ ​be​ ​used​ ​for​ ​application​ ​development.


## Problem​ ​Statement 
#### The problem to be solved in this project is to predictthefuturestockpricesfornextcouple(5,10,20)of                      days.  This problem will be structured as a supervised learning problem where the input dataset willbeframed                 in such awaythatitwillhaveinputandmatchingoutputvalues.Aslidingwindowtechniquewillbeused                    to traverse input dataset and matching output. This will be considered a regression type of supervised                learning​ ​as​ ​the​ ​output​ ​will​ ​be​ ​a​ ​continuous​ ​real​ ​value. The input dataset will bethehistoricaldailyprices(ofthechosenStocksymbol)datasetdownloadedfrom                the Yahoo finance website​8​. Adjusted Close price for each day will be considered as an input.

#### Theinput                  values indexed on date will be treated as a time series(sequence). This dataset will be split intotraining,                  validation and test datasets. The training dataset will further be traversed using a sliding window               technique. A sliding window with a fixed number of Adj Close values will betheinputvaluesandfixed                   number( depending upon the number of values to be predicted) of values immediately after the sliding                window in sequence will be the matching output values. Validation dataset will be used to test model                 during​ ​training​ ​and​ ​Testing​ ​dataset​ ​will​ ​be​ ​used​ ​to​ ​test​ ​the​ ​final​ ​model. 

#### A model will be developed matching input window values to the output values using Deep Learning.                After developing the model using training dataset, stock price will be predicted for the period starting                where the training dataset period ends. The predicted values can be compared against the testing dataset                values for evaluation. The closeness of predicted values to actual values can be measured using Kera’s                evaluate​ ​method​ ​or​ ​using​ ​RMSE​ ​method. 

## Datasets​ ​and​ ​Inputs 
#### Data for the project is taken from Yahoo finance website​8​. Historical data of following stock tickers is                 taken​ ​–​ ​S&P500,​ ​MSFT,​ ​XOM,​ ​MYL,​ ​WMT,​ ​PFE,​ ​IBM,​ ​AAPL,​ ​GOOG. All available historical data for these stocks from yahoo finance website​8 ​will be taken. Different stocks                have​ ​different​ ​periods​ ​of​ ​data​ ​available. 

#### The historical data will be collected for following events –Open,Close,High,Low,AdjClose,Volume.                 Here, we will use only Adj Close​5 price ofthestocks.AdjClosepriceissimilartodailycloseprice,only                     difference​ ​is​ ​that​ ​it​ ​takes​ ​care​ ​of​ ​past​ ​splits,​ ​reverse​ ​splits,​ ​dividends,​ ​rights​ ​offerings​ ​also. 

#### Final dataset willcontainAdjClosecolumnwithDateastheindex.Datasetwillbecleanedforany“null”                   values and it will be made sure that Adj Close data only have float values. Pandas dataframe dropna                  function​ ​will​ ​be​ ​used​ ​to​ ​drop​ ​any​ ​rows​ ​containing​ ​undefined​ ​data. 

#### After data cleanup, it will be normalized so that its values are scaled between -1 and 1. It will help in                     convergence of machine learning algorithm and it will also help in bringing different stocks to thesame                 scale.  

#### Datasets​ ​for​ ​each​ ​stock​ ​will​ ​be​ ​split​ ​into​ ​training​ ​and​ ​test​ ​datasets​ ​in​ ​the​ ​ratio​ ​of​ ​60/40​ ​or​ ​80/20. 

#### A fixed window of data values from sequence will be taken as inputs and theafixednumber(depending                  upon the number ofpredictedvalues)ofvaluesjustaftertheinputwindowistakenasoutputs.Windowis                   rolled​ ​forward​ ​repeatedly​ ​to​ ​traverse​ ​the​ ​input​ ​dataset​ ​to​ ​get​ ​inputs​ ​and​ ​matching​ ​outputs. 

#### Although chances of testing data information being trickled to training are negligible, as testing is done                after​ ​model​ ​is​ ​fully​ ​trained​ ​using​ ​Keras.

#### To​ ​avoid​ ​testing​ ​dataset​ ​information​ ​trickle​ ​down​ ​to​ ​training​ ​dataset​ ​
 #### ● Testing​ ​data​ ​is​ ​used​ ​after​ ​model​ ​is​ ​fully​ ​trained 
 #### ● Training​ ​and​ ​testing​ ​will​ ​be​ ​done​ ​on​ ​different​ ​stock​ ​symbols. 
 
## Solution​ ​Statement 
#### In this project, attempt will be made to predict the prices using Deep Learning techniques of Recurrent                 Neural Networks. Recurrent Neural Network model LSTM (Long short term Memory) will be used to               predict​ ​the​ ​stock​ ​prices.  
 
 
## Evaluation​ ​Metrics 
#### Predicted​ ​prices​ ​will​ ​be​ ​compared​ ​to​ ​Actual​ ​historical​ ​prices. 
 #### ● Keras​ ​model.evaluate​ ​method​ ​will​ ​be​ ​used​ ​to​ ​find​ ​the​ ​training​ ​and​ ​test​ ​errors. 
 #### ● MSE​ ​(Mean​ ​Square​ ​Error)​ ​will​ ​be​ ​calculated​ ​for​ ​predicted​ ​and​ ​actual​ ​values. 
#### As this project involves comparing two number series, MSE seems to be an appropriate              function​ ​to​ ​compare​ ​values.​ ​It​ ​is​ ​one​ ​of​ ​the​ ​simpler​ ​methods​ ​to​ ​compare​ ​series. 
 
 
## Analysis 
#### Project is implemented using programming language Python and neural network API Keras is used for               deep​ ​learning. 

## Benchmark​ ​Model 
#### Multiple​ ​benchmark​ ​models​ ​will​ ​be​ ​used​ ​for​ ​comparison: 
#### 1) As we are using existing data and we have test data available for which actual values are already                  available. We can test our model generated predicted values with these actual values and can see how accurate​ ​our​ ​model​ ​is. 

#### 2) Use​ ​Linear​ ​Regression​ ​prediction​ ​Model
 
#### 3) Find\develop prediction model using KNN(K nearest neighbor) regression model and run the model             on​ ​same​ ​data​ ​as​ ​to​ ​be​ ​used​ ​on​ ​Deep​ ​Learning​ ​model.​ ​KNN​ ​model​ ​will​ ​be​ ​used​ ​as​ ​a​ ​benchmark. 
 
#### 4) Use​ ​Support​ ​vector​ ​Machine(​ ​RBF​ ​and​ ​Poly)​ ​methods 
>>>>>>> 7f75f26... added capstone project
=======
# Machine​ ​Learning​ ​Engineer​ ​Nanodegree
## Capstone​ ​Project
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

## Predict​ ​Stock​ ​Prices​ ​using​ ​Deep​ ​Learning

### Navneet​ ​Latawa
### 28​ ​Sep​ ​2017

## Definition

#### In this MLND capstone project, I will be attempting to predict future stock prices for selective stock symbols. I will take historical stock price data for specific Stock symbols from yahoo finance and use deep learning techniques to train a model. This model will then be used to predict prices for next couple of​ ​days. As a reference, I will be using Machine Learning techniques taught in Udacity’s Machine Learning NanoDegree​ ​and​ ​Machine​ ​Learning​ ​for​ ​Trading​ ​course. Python​ ​language​ ​and​ ​Keras​​ library​ ​will​ ​be​ ​used​ ​for​ ​application​ ​development.


## Problem​ ​Statement
#### The problem to be solved in this project is to predictthefuturestockpricesfornextcouple(5,10,20)of                      days.  This problem will be structured as a supervised learning problem where the input dataset willbeframed                 in such awaythatitwillhaveinputandmatchingoutputvalues.Aslidingwindowtechniquewillbeused                    to traverse input dataset and matching output. This will be considered a regression type of supervised                learning​ ​as​ ​the​ ​output​ ​will​ ​be​ ​a​ ​continuous​ ​real​ ​value. The input dataset will bethehistoricaldailyprices(ofthechosenStocksymbol)datasetdownloadedfrom                the Yahoo finance website​8​. Adjusted Close price for each day will be considered as an input.

#### Theinput                  values indexed on date will be treated as a time series(sequence). This dataset will be split intotraining,                  validation and test datasets. The training dataset will further be traversed using a sliding window               technique. A sliding window with a fixed number of Adj Close values will betheinputvaluesandfixed                   number( depending upon the number of values to be predicted) of values immediately after the sliding                window in sequence will be the matching output values. Validation dataset will be used to test model                 during​ ​training​ ​and​ ​Testing​ ​dataset​ ​will​ ​be​ ​used​ ​to​ ​test​ ​the​ ​final​ ​model.

#### A model will be developed matching input window values to the output values using Deep Learning.                After developing the model using training dataset, stock price will be predicted for the period starting                where the training dataset period ends. The predicted values can be compared against the testing dataset                values for evaluation. The closeness of predicted values to actual values can be measured using Kera’s                evaluate​ ​method​ ​or​ ​using​ ​RMSE​ ​method.

## Datasets​ ​and​ ​Inputs
#### Data for the project is taken from Yahoo finance website​8​. Historical data of following stock tickers is                 taken​ ​–​ ​S&P500,​ ​MSFT,​ ​XOM,​ ​MYL,​ ​WMT,​ ​PFE,​ ​IBM,​ ​AAPL,​ ​GOOG. All available historical data for these stocks from yahoo finance website​8 ​will be taken. Different stocks                have​ ​different​ ​periods​ ​of​ ​data​ ​available.

#### The historical data will be collected for following events –Open,Close,High,Low,AdjClose,Volume.                 Here, we will use only Adj Close​5 price ofthestocks.AdjClosepriceissimilartodailycloseprice,only                     difference​ ​is​ ​that​ ​it​ ​takes​ ​care​ ​of​ ​past​ ​splits,​ ​reverse​ ​splits,​ ​dividends,​ ​rights​ ​offerings​ ​also.

#### Final dataset willcontainAdjClosecolumnwithDateastheindex.Datasetwillbecleanedforany“null”                   values and it will be made sure that Adj Close data only have float values. Pandas dataframe dropna                  function​ ​will​ ​be​ ​used​ ​to​ ​drop​ ​any​ ​rows​ ​containing​ ​undefined​ ​data.

#### After data cleanup, it will be normalized so that its values are scaled between -1 and 1. It will help in                     convergence of machine learning algorithm and it will also help in bringing different stocks to thesame                 scale.  

#### Datasets​ ​for​ ​each​ ​stock​ ​will​ ​be​ ​split​ ​into​ ​training​ ​and​ ​test​ ​datasets​ ​in​ ​the​ ​ratio​ ​of​ ​60/40​ ​or​ ​80/20.

#### A fixed window of data values from sequence will be taken as inputs and theafixednumber(depending                  upon the number ofpredictedvalues)ofvaluesjustaftertheinputwindowistakenasoutputs.Windowis                   rolled​ ​forward​ ​repeatedly​ ​to​ ​traverse​ ​the​ ​input​ ​dataset​ ​to​ ​get​ ​inputs​ ​and​ ​matching​ ​outputs.

#### Although chances of testing data information being trickled to training are negligible, as testing is done                after​ ​model​ ​is​ ​fully​ ​trained​ ​using​ ​Keras.

#### To​ ​avoid​ ​testing​ ​dataset​ ​information​ ​trickle​ ​down​ ​to​ ​training​ ​dataset​ ​
 #### ● Testing​ ​data​ ​is​ ​used​ ​after​ ​model​ ​is​ ​fully​ ​trained
 #### ● Training​ ​and​ ​testing​ ​will​ ​be​ ​done​ ​on​ ​different​ ​stock​ ​symbols.

## Solution​ ​Statement
#### In this project, attempt will be made to predict the prices using Deep Learning techniques of Recurrent                 Neural Networks. Recurrent Neural Network model LSTM (Long short term Memory) will be used to               predict​ ​the​ ​stock​ ​prices.  


## Evaluation​ ​Metrics
#### Predicted​ ​prices​ ​will​ ​be​ ​compared​ ​to​ ​Actual​ ​historical​ ​prices.
 #### ● Keras​ ​model.evaluate​ ​method​ ​will​ ​be​ ​used​ ​to​ ​find​ ​the​ ​training​ ​and​ ​test​ ​errors.
 #### ● MSE​ ​(Mean​ ​Square​ ​Error)​ ​will​ ​be​ ​calculated​ ​for​ ​predicted​ ​and​ ​actual​ ​values.
#### As this project involves comparing two number series, MSE seems to be an appropriate              function​ ​to​ ​compare​ ​values.​ ​It​ ​is​ ​one​ ​of​ ​the​ ​simpler​ ​methods​ ​to​ ​compare​ ​series.


## Analysis
#### Project is implemented using programming language Python and neural network API Keras is used for               deep​ ​learning.

## Benchmark​ ​Model
#### Multiple​ ​benchmark​ ​models​ ​will​ ​be​ ​used​ ​for​ ​comparison:
#### 1) As we are using existing data and we have test data available for which actual values are already                  available. We can test our model generated predicted values with these actual values and can see how accurate​ ​our​ ​model​ ​is.

#### 2) Use​ ​Linear​ ​Regression​ ​prediction​ ​Model
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

#### 3) Find\develop prediction model using KNN(K nearest neighbor) regression model and run the model             on​ ​same​ ​data​ ​as​ ​to​ ​be​ ​used​ ​on​ ​Deep​ ​Learning​ ​model.​ ​KNN​ ​model​ ​will​ ​be​ ​used​ ​as​ ​a​ ​benchmark.

#### 4) Use​ ​Support​ ​vector​ ​Machine(​ ​RBF​ ​and​ ​Poly)​ ​methods
<<<<<<< HEAD
=======

#### 3) Find\develop prediction model using KNN(K nearest neighbor) regression model and run the model             on​ ​same​ ​data​ ​as​ ​to​ ​be​ ​used​ ​on​ ​Deep​ ​Learning​ ​model.​ ​KNN​ ​model​ ​will​ ​be​ ​used​ ​as​ ​a​ ​benchmark.

#### 4) Use​ ​Support​ ​vector​ ​Machine(​ ​RBF​ ​and​ ​Poly)​ ​methods

>>>>>>> 0409cfc... readme.md updated
=======

>>>>>>> 41dfc39... Create readme.md
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
