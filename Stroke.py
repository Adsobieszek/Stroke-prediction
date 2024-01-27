import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data = data.drop(columns=['id'])
#data description:
#    id: Unique identifier for each individual.
#   gender: The gender of the individual (Male / Female / Other).
#    age: Age of the individual. (range 0-82 years)
#    hypertension: Whether the individual has hypertension (1 for yes, 0 for no).
#    heart_disease: Whether the individual has heart disease (1 for yes, 0 for no).
#    ever_married: Whether the individual is ever married (e.g., yes or no).
#    work_type: The type of work the individual (Private / Self-employed / Got_job / Children / Never_worked).
#    Residence_type: The type of residence (Urban / Rural).
#    avg_glucose_level: The average glucose level in the individual's blood.
#    bmi: Body Mass Index (a measure of body fat based on height and weight).
#    smoking_status: The smoking status of the individual (formerly smoked / never smoked / smokes / Unknown).
# #    stroke: The target variable indicating whether the individual had a stroke (1 for yes, 0 for no).


# # #checking if there is some empty data
missing_values = data.isnull().sum()
print(missing_values)

# # # #there are some rows where bmi cell is empty so we removing them since I have enough data
data.dropna(subset=['bmi'], inplace=True)

# Assuming you have a dataset with a column named 'work_type' and 'smoking_status'

# Convert work_type to binary indicators
data['private_work'] = (data['work_type'] == 'Private').astype(int)
data['self_employed'] = (data['work_type'] == 'Self-employed').astype(int)
data['got_job'] = (data['work_type'] == 'Got_job').astype(int)
data['children'] = (data['work_type'] == 'Children').astype(int)
data['never_worked'] = (data['work_type'] == 'Never_worked').astype(int)

# Convert smoking_status to binary indicators
data['formerly_smoked'] = (data['smoking_status'] == 'formerly smoked').astype(int)
data['never_smoked'] = (data['smoking_status'] == 'never smoked').astype(int)
data['smokes'] = (data['smoking_status'] == 'smokes').astype(int)
data['unknown_smoking_status'] = (data['smoking_status'] == 'Unknown_smoke').astype(int)


data.drop(columns=['work_type', 'smoking_status'], inplace=True)


# # # #we can see that there are almonst no data where gender = other, so we are removing rows and changing it to binary values
data = data[data['gender'] != 'Other']
data["gender"] = data["gender"].replace(["Male", "Female"], [1, 0])

data["ever_married"] = data["ever_married"].replace(["Yes", "No"], [1, 0])
data["Residence_type"] = data["Residence_type"].replace(["Urban", "Rural"], [1, 0])

print(data.info())
print(data.head())

categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'Private','Self-employed','Got_job','Children','Never_worked','formerly smoked',
'never smoked','smokes','Unknown_smoke']
numerical = ['age','avg_glucose_level', 'bmi']




# # Correlation Matrix
correlation_matrix=data[['age','avg_glucose_level','bmi','stroke']].corr()


# # Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

#Bar Chart
class_counts = data['stroke'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution (stroke vs. no-stroke)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0, 1], ['No Stroke', 'Stroke'])  
plt.show()






# ### 2 czesc kodu

X=data.drop('stroke',axis=1)
y=data['stroke']
# Resampling z użyciem SMOTE (undersampling)
from imblearn.under_sampling import RandomUnderSampler
sampler=RandomUnderSampler(random_state=42,replacement=True)
X_sampl,y_sampl=sampler.fit_resample(X,y)

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_sampl, y_sampl, test_size=0.2, random_state=1)

#data normalization 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing dataabs
X_test_norm = norm.transform(X_test)

print(X_train_norm)


#model SVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1.0, loss='hinge'))
])


svm_clf.fit(X_train_norm, y_train)

# Predykcja na zestawie testowym
y_pred = svm_clf.predict(X_test_norm)
print(y_pred)
# Wydruk wyników
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy SVM:', accuracy_score(y_test, y_pred))
print('Precision SVM:', precision_score(y_test, y_pred))
print('Recall SVM:', recall_score(y_test, y_pred))
print('F1 Score SVM:', f1_score(y_test, y_pred))
print("\n","\n")

from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_test,y_pred)

def plotRocCurve(fpr,tpr):
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'r')
plotRocCurve(fpr, tpr)
plt.title('ROC Curve')
#plt(plotRocCurve(fpr, tpr))

##### KNN CLASSIFIER

from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_norm,y_train)
y_pred=knn_clf.predict(X_test_norm)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("KNN 3 BEFORE SCALING:","\n")
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print("\n","\n")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

Knn_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())])


X_train_norm=Knn_pipe.fit_transform(X_train_norm)
X_test_norm=Knn_pipe.fit_transform(X_test_norm)
knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_norm,y_train)

y_pred=knn_clf.predict(X_test_norm)

print("KNN 3 AFTER SCALING:","\n")
print('Accuracy:', accuracy_score(y_test, y_pred).round(2))
print('Precision:', precision_score(y_test, y_pred).round(2))
print('Recall:', recall_score(y_test, y_pred).round(2))
print('F1 Score:', f1_score(y_test, y_pred).round(2))
print("\n","\n")

from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_norm,y_train)
y_pred=knn_clf.predict(X_test_norm)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("KNN 5 BEFORE SCALING:","\n")
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print("\n","\n")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

Knn_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())])


X_train_norm=Knn_pipe.fit_transform(X_train_norm)
X_test_norm=Knn_pipe.fit_transform(X_test_norm)
knn_clf=KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_norm,y_train)

y_pred=knn_clf.predict(X_test_norm)

print("KNN 5 AFTER SCALING:","\n")
print('Accuracy:', accuracy_score(y_test, y_pred).round(2))
print('Precision:', precision_score(y_test, y_pred).round(2))
print('Recall:', recall_score(y_test, y_pred).round(2))
print('F1 Score:', f1_score(y_test, y_pred).round(2))
print("\n","\n")

### DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
tree_clf=DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train,y_train)

from sklearn.tree import plot_tree
plot_tree(tree_clf)




y_pred=tree_clf.predict(X_test)
from sklearn.metrics import classification_report
print("classification report of decision tree:","\n","\n",classification_report(y_test,y_pred))


#UNSUPERVISED MACHINE LEARNING METHODS

#DBSCAN METHOD


features_clus=['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type']
X_cluster = X_sampl[features_clus].values

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)

clusters = dbscan.fit_predict(X_cluster)


df_encoded_with_clusters_dbscan = X_sampl
df_encoded_with_clusters_dbscan['dbscan_cluster'] = clusters

unique_clusters = set(clusters)
print(f"Unique clusters (including noise): {unique_clusters}")
print(f"Number of noise points: {list(clusters).count(-1)}")





################    K-MEANS
from sklearn.cluster import KMeans
X_cluster = X_sampl[['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type']]

n_cluster = range(1, 21)
kmeans = [KMeans(n_clusters=i).fit(X_cluster) for i in n_cluster]
scores = [kmeans[i].score(X_cluster) for i in range(len(kmeans))]

f = plt.figure(1, figsize=(16,6))
plt.plot(scores)
plt.xticks(n_cluster)
import numpy as np
scores = np.array(scores)
dif_scores = scores / scores[0]
dif_scores = np.diff(dif_scores)
n_clusters = np.argwhere(dif_scores < np.quantile(dif_scores, 0.9))[-1][0]
print(n_clusters)


#n_clusters=17
# Applying KMeans Clustering
kmeans = KMeans(n_clusters=17, random_state=42)
X_sampl['stroke_cluster'] = kmeans.fit_predict(X_cluster)
df_encoded_with_clusters_kmeans = X_sampl

plt.figure(figsize=(10, 6))
sns.countplot(x='stroke_cluster',hue=y_sampl, data=df_encoded_with_clusters_kmeans)
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
#plt.show()
