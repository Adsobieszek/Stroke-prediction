# Stroke prediction

The significant percentage of strokes are preventable. In this project, my goal is to delve into the world of stroke prediction. The dataset I'm working with spans diverse age groups, genders, and includes information about habits and various health-related factors. Ultimately, the objective is to build a predictive model. Through transformations and selecting the best model, I aspire to develop an effective tool that can estimate the probability of stroke based on the provided data. 

# Data cleaning and visualizations
<img src="1.png" width="600">

Firstly, i tried to clean the data. I made some steps, like dropping 'id' columns which was not useable,  removing rows where 'bmi' was null since i knew that i can afford that due to having enough data. I've converted 'work_type' and 'smoking_status' to binary indictaros. I also realised that there were almost no data where gender = other, so i have removed those rows and changed it to binary values.

<img src="2.png" width="600">
Then i created matrix correlation with numeric variables and i turned out that the correlation between all variables is small so i could take it as it was.

<img src="3.png" width="600">

Histogram & density plots describes the age, average glucose level and bmi. The median group are adult people. Glucose level distribution is skewed towards left and the biggets group has around 80 mg/dl. 
BMI is highly skewed and the median is around 27.

<img src="5.png" width="600">

The plot displays average glucose level and age. We can see the trend, the more aged people and higher glucose level, then more stroke events.

<img src="6.png" width="600">

This plot  displays average glucose level and bmi. It is similiar to previous one, stroke occurs more to people with high bmi and high glucose level.

<img src="7.png" width="600">

# Building models
Our data was imbalanced, that could have big impact on the model performance. So i decided to use undersampling method - SMOTE. 

<img src="4.png" width="600">
<img src="8.png" width="600">

Then i moved to split data to train and test sets. I have normalized data, using MinMaxScaler. Having features on similar scales can prevent one feature from dominating the learning process simply because it has a larger magnitude. The first algorithm I used was belonging to Supervised Machine Learning - Support Vector Machine. SVM is algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes, maximizing the margin between classes and relying on support vectors for decision boundary determination.

<img src="9.png" width="600">
The scores of the SVM algorithm:

- Accuracy (0.83): This represents the overall correctness of the model in predicting both stroke and non-stroke cases. An accuracy of 83% suggests that the model correctly predicted 83% of the cases.

- Precision (0.93): This indicates the proportion of predicted stroke cases that are true positives. In the context of a stroke predictor, high precision means that when the model predicts a person is at risk of a stroke, it is correct about 93% of the time.

- Recall (0.78): This measures the model's ability to correctly identify individuals who are actually at risk of a stroke. A recall of 78% suggests that the model captures about 78% of the true stroke cases.

- F1 Score (0.84):  It provides a balanced measure, especially useful when false positives and false negatives are both critical. An F1 Score of 84% indicates a good balance between precision and recall.

In the context of a stroke predictor, achieving high precision is essential to minimize false alarms, while a high recall ensures that the model identifies a significant portion of actual stroke cases. It's crucial to consider the specific requirements of the task and the potential impact of false positives and false negatives in a medical context.

<img src="10.png" width="600">


Then, i used KNN algorithms with k=3 and k=5. K-nearest neighbors, is a classification algorithm that assigns a data point to a specific category based on the majority class of its k-nearest neighbors in the feature space. It utilizes the principle that similar data points tend to belong to the same class, making it a non-parametric and instance-based learning method in machine learning.

<img src="11.png" width="600">

This model knn=5 is more balanced, more resistant to noise or deviations in the data than knn=3 because it can be used for more advanced forecasts.

<img src="12.png" width="600">
<img src="13.png" width="600">
<img src="14.png" width="600">
<img src="15.png" width="600">
<img src="16.png" width="600">
