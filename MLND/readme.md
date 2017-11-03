# Udacity Machine Learning Nanodegree Projects

 This repository contains project files for Udacity's Machine Learning Engineer Nanodegree program(https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009)

## Projects

### Titanic Survival Exploration [Exploratory Project]
 In this introductory project, a subset of the RMS Titanic passenger manifest are explored to determine which features best predict whether someone survived or did not survive. 


### Predicting Boston Housing Prices - [Model Evaluation & Validation]
 In this project, a model is built to predict the value of a given house in the Boston real estate market using various statistical analysis tools. 
 The Boston housing dataset for this project originated from the UCI Machine Learning Repository. DecisionTreeRegressor from sklearn library is 
 used to train the model. sklearn grid_search is used to find optimal parameters.

### Student Intervention System [Supervised Learning]
 The goal for this project is to identify students who might need early intervention before they fail to graduate
 Models tried
	- Logistic Regression
	- GaussianNB
	- SVM
 I have chosen these three algorithms, as we have small dataset\training set size, need classification algorithm, and need to explain the model selection to the management. Other algorithms like Ensemble, Neural networks are difficult to explain to management.
 Althought Decision trees may be a good option for explanation to managemebt, but as number of features are about 48 (31 originally) and dataset size is small, so Decision trees are also not a good option.
 Best Model Selection - If we consider Training and Testing time, then GaussianNB is the best method and SVM is the worst method. But if we consider F1_score wise, then Logistic Regression(LR) is the best and GaussianNB is the worst.
 SVM seems to overfit also as training F1_Score seems to increase but Testing F1_score seems to be falling.
 Although execution time for LR is in-between GaussianNB and SVM, but its performance is much better than the both. Although training F1_score of LR is worse than SVM but test F1_score of LR is better.So I will consider Logistic Regression as the best model for this dataset.

### Finding Donors for CharityML [Supervised Learning]
 In this project, several supervised algorithms are used to accurately model individuals' income using data collected from the 1994 U.S. Census. 
 The dataset for this project originates from the UCI Machine Learning Repository. 
 Models tried
	- Logistic Regression
	- GaussianNB
	- SVM
	- Decision Trees
	- Random Forest
	- AdaBoost
	
 Although six models are explored but I think three best models to explain Donors are LogisticRegression, SVC, DecisiontreeClassifier.	
 LogisticRegression is chosen because:
	- good performance( Accuracy score, F-score) on training and test datasets
	- very low time on training data, low time on testing data
	- easy to explain to management

 SVM is chosen because:
	- good performance( Accuracy score, F-score) on training and test datasets - comparable to Logistical regression
	- very high training and testing time
	- easy to explain to management

 DecisionTree is chosen because:
	- good performance( Accuracy score, F-score) on training and test datasets - better than LR and SVC but low compared to AdaBoost and RandomForest
	- low traning and testing time
	- easy to explain to management

 Final model chosen is DecisionTreeClassifier as out of the three( LR, SVC, DTC) selected above, DTC has best performance scores and comparable training\testing times.
 DTC is also easier to explain seeing the tree node options.
 SVC is rejected mainly because of high training and testing times.	


### Creating Customer Segments [Unsupervised Learning]
 In this project, a dataset containing data on various customers' annual spending amounts (reported in monetary units) of diverse product categories for internal structure is analyzed. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
 The dataset for this project can be found on the UCI Machine Learning Repository. 
 A scatter matrix of each of the six product  features is plotted to identify the correlations between features so that the correlated features can be excluded from further analysis.
 PCA (Principal Component Analysis) is used to identify the best features - after this analyis 4 product features were used instead of total 6 input features
 Finally K-means clustering was used to cluster the customers.

### Train a Smartcab to Drive [ Reinforcement Learning]
 In this project, optimized Q-Learning driving agent is built to navigate a Smartcab through its environment towards a goal. Since the Smartcab is expected to drive passengers from one location to another, the driving agent will be evaluated on two very important metrics: Safety and Reliability. A driving agent that gets the Smartcab to its destination while running red lights or narrowly avoiding accidents would be considered unsafe. Similarly, a driving agent that frequently fails to reach the destination in time would be considered unreliable. Maximizing the driving agent's safety and reliability would ensure that Smartcabs have a permanent place in the transportation industry.

### Dog Identification\Classification (Deep Learning - Convolutional Neural Networks)
 In this project, CNN deep  learning model is developed to identify dog breeds in images. It will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 
 Haar feature-based cascade classifiers are used to identify human faces in images.
 
### InterviewQuestions
#### This project required implementation of five algorithms using python. It also involved explaining the algorithm design, time and space complexity.
	- Determine whether some anagram of t is a substring of safety
	- Find the longest palindromic substring contained in a string
	- Find the minimum spanning tree within a graph
	- Find the least common ancestor between two nodes on a binary search tree
	- Find the element in a singly linked list that's m elements from the end

### Machine Learning Capstone Project - Predict Stock Prices [ Deep Learning - LSTM]
 In this MLND capstone project, I attempted to predict future stock prices for selective stock symbols. I took historical stock price data for specific Stock symbols from yahoo finance and used    deep learning LSTM model for training. This model was then  used to predictprices for next couple of​ ​days. 