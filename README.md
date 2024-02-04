# Stroke-prediction

Do you know, 80% Heart strokes are preventable?, yes they are. In this notebook, we attempted to visulize some key indicators that lead to heart strokes. Here data is sampled from a wide range of age groups, gender, habits and health related issues. Most of the visualizations are self expilantory and try to stick to simple visualization but effiective method to convey most of the information.
Problem Statement

Visululize the relationships between various Healthy and Unhealthy habits to Heart Strokes, and there by predict the stroke probability with best model and hypertuned parameters.

<img src="4.png" width="1000">
# Data cleaning
 ![1](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/c4ff14f2-bd70-478d-9600-e8bb8968fc1b)

Firstly, i tried to clean the data. I made some steps, like dropping 'id' columns which was not useable,  removing rows where 'bmi' was null since i knew that i can afford that due to having enough data. I've converted 'work_type' and 'smoking_status' to binary indictaros. I also realised that there were almost no data where gender = other, so i have removed those rows and changed it to binary values.

![2](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/f957fd84-b61f-4a0e-95e9-a3634bb70370)

Then i created matrix correlation with numeric variables and i turned out that the correlation between all variables is small so i could take it as it was.

![3](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/92c2e65e-3614-4261-8b3d-30de2760b174)

Histogram & density plots describes the age, average glucose level and bmi. The median group are adult people. Glucose level distribution is skewed towards left and the biggets group has around 80 mg/dl. 
BMI is highly skewed and the median is around 27.

![5](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/37d91426-c4a8-440b-bf43-be3bac7500a3)

The plot displays average glucose level and age. We can see the trend, the more aged people and higher glucose level, then more stroke events.

![6](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/9f5cf842-21ef-41b5-9c5b-c04270e8195d)

This plot  displays average glucose level and bmi. It is similiar to previous one, stroke occurs more to people with high bmi and high glucose level.

![7](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/eebe50dc-d583-4482-bf7a-7dc84379fea0)

Our data was imbalanced, that could have big impact on the model performance. So i decided to use undersampling method - SMOTE. 

![4](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/cb29817a-3c66-42a1-96a0-6a10a9720887)
![8](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/38ebd536-f49a-4f2a-86c0-62187a1df380)

Then i moved to split data to train and test sets. I have normalized data, using MinMaxScaler. Having features on similar scales can prevent one feature from dominating the learning process simply because it has a larger magnitude. The first algorithm I used was belonging to Supervised Machine Learning - Support Vector Machine. SVM is algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes, maximizing the margin between classes and relying on support vectors for decision boundary determination.

![9](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/0d6ebdd9-d12d-4a52-a495-08223b0048ea)

The scores of the SVM algorithm:

- Accuracy (0.83): This represents the overall correctness of the model in predicting both stroke and non-stroke cases. An accuracy of 83% suggests that the model correctly predicted 83% of the cases.

- Precision (0.93): This indicates the proportion of predicted stroke cases that are true positives. In the context of a stroke predictor, high precision means that when the model predicts a person is at risk of a stroke, it is correct about 93% of the time.

- Recall (0.78): This measures the model's ability to correctly identify individuals who are actually at risk of a stroke. A recall of 78% suggests that the model captures about 78% of the true stroke cases.

- F1 Score (0.84):  It provides a balanced measure, especially useful when false positives and false negatives are both critical. An F1 Score of 84% indicates a good balance between precision and recall.

In the context of a stroke predictor, achieving high precision is essential to minimize false alarms, while a high recall ensures that the model identifies a significant portion of actual stroke cases. It's crucial to consider the specific requirements of the task and the potential impact of false positives and false negatives in a medical context.


![10](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/1b9c0881-5c33-4999-861f-fa23eee1c55d)
![11](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/5af17f3e-cd52-4b05-a7c4-1c33f14e7cf2)
![12](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/d366c042-b38f-4e44-9d28-27303edaf798)
![13](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/6ffdf00c-3836-48b4-9ec5-95f6fd6ae0c0)
![14](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/11492f38-28c4-45f3-bd73-59e75edde578)
![15](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/9d626e9a-0634-4c7f-8456-874bcde4cd97)
![16](https://github.com/Adsobieszek/Stroke-prediction/assets/134363499/41383527-735f-4536-b4d8-a50375f62e3d)
