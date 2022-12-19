# Loading python packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

#Loading online news popularity data
onlinenews = pd.read_csv('C:\\Users\\ashok\\Desktop\\OnlineNewsPopularity.csv')


print(onlinenews.head())

#Identifying categorical and continuos variables
categorical = onlinenews.select_dtypes(include=['object'])

for i in categorical:
    column = categorical[i]
    print('Variable: {} with {} unique values '.format(i,column.nunique()))

#Remove the only categorical variable, it is not needed in this analysis
onlinenews = onlinenews.drop('url', axis =1)

#Looking for null values
print('All null values are in yellow')
sns.heatmap(onlinenews.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show() #Plot shows that there is no null value at all

# Function to create a list of variables with outliers
def outliersinColumns(df):

    columns_outliers = []
    for column in df.columns:

        if onlinenews[column].nunique() > 2:  # Apply for variables with 3 or more unique values

            for value in df[column]:
                if value:
                    columns_outliers.append(column)
                    break

    return columns_outliers

#Using a statistical method to detect outliers: interquartile range (IQR)

Q1 = onlinenews.quantile(0.25)
Q3 = onlinenews.quantile(0.75)
IQR = Q3 - Q1


#Creating notinvalidarea dataframe with boolean values:
#False means these values are into the valid area
#True indicates presence of an outlier
notinvalidarea = (onlinenews < (Q1 - 1.5 * IQR)) | (onlinenews > (Q3 + 1.5 * IQR))

#Calling function outliersinColumns
columns_w_outliers = outliersinColumns(notinvalidarea)

#Printing Results
print('Columns with outliers: {}'.format(len(columns_w_outliers)))
print('\n')
print(columns_w_outliers)

#Function to remove outliers
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


for col1 in onlinenews.columns:
    for col2 in columns_w_outliers:
        if col1 == col2:
            processed_df = remove_outlier(onlinenews,col2) # storing the data with outlier removal into new dataframe

print('after outliers removal, all Null values are in yellow')
sns.heatmap(processed_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


#Check the correlation
correlation = processed_df.corr()
plt.figure(figsize=(25,25))
sns.heatmap(correlation, square=True, annot=True, linewidths=.5)
plt.title("Correlation Matrix (Online News)")
plt.show()

print(correlation.head(5))

processed_df = processed_df.rename(columns=lambda x: x.split(" ")[1])


#This is a classification problem, so it is required to classify the target attribute 'shares' into 3 groups
#3 groups: 'Low','Medium','High'

processed_df['labelbin_shares'] = pd.cut(processed_df.shares, bins=[0,946,2800,843300], labels=['Low','Medium','High'])
processed_df['bin_shares'] = pd.cut(processed_df.shares, bins=[0,946,2800,843300], labels=[1,2,3])

# percentage and number of shares for all 3 groups
print(processed_df['bin_shares'].value_counts()/processed_df['labelbin_shares'].count())
print(processed_df['labelbin_shares'].value_counts())

#Univariate Feature Selection
# Select features according to the k highest scores using SelectKBest

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# X = set of all input data attributes and y = output attribute 'shares'
X = processed_df.drop(columns=['labelbin_shares','shares','bin_shares'])
y = processed_df['bin_shares']

# Create an SelectKBest object to select features with two best ANOVA F-Values
f_values_selection = SelectKBest(f_classif, k=21)

# Apply the SelectKBest object to the features and target
KBest_in_X = f_values_selection.fit_transform(X, y)

print('Original number of features:', X.shape[1])
print('Reduced number of features:', KBest_in_X.shape[1])
print('\n')

bool_masking = f_values_selection.get_support() #list of booleans
Best_Selected_Features = [] # The list of your K best features

for boolean, features in zip(bool_masking, X.columns.values):
    if boolean:
        Best_Selected_Features.append(features)

print(Best_Selected_Features)

#Check feature importance using Random Forest

X = processed_df[Best_Selected_Features]
Y = processed_df['bin_shares']

rfc = ensemble.RandomForestClassifier(n_estimators=100)

#Fitting the model
rfc.fit(X,Y)

Important_Features = {}
for features,importance in zip(Best_Selected_Features,rfc.feature_importances_):
    Important_Features[features] = importance
    #print(feature,importance)

plt.barh(Best_Selected_Features,rfc.feature_importances_,height=.5)
plt.show()

#After running a random forest and based on features importance, less importance features will be removed from selectedFeatures

to_be_removed = ['data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world','weekday_is_saturday','weekday_is_sunday']
for remove in to_be_removed:
    Best_Selected_Features.remove(remove)

from collections import Counter
from imblearn.combine import SMOTEENN

X = processed_df[Best_Selected_Features]
y = processed_df['bin_shares']

#SMOTEENN, it's a method that combines over-sampling and under-sampling.
#It's a class to perform over-sampling using SMOTE and cleaning using ENN.

smote = SMOTEENN(random_state=42)
X_final, y_final = smote.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_final))
# putting target labels for classification report
target_labels = ['Low','Medium','High']

# 1. Random Forest Modelling

X = X_final
Y = y_final

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=3000,
                       n_jobs=None, oob_score=False)

#Fitting the model
rfc.fit(X_train,y_train)

#Making predictions
rfc_predictions = rfc.predict(X_test)

#Get results with metrics.accuracy_score or rfc.score(X_test, y_test)
rfc_accu_score = round(metrics.accuracy_score(y_test, rfc_predictions),4)
print('\n\n')
print('ACCURACY FOR RANDOM FOREST :{}'.format(rfc_accu_score))

#Classification Report
target_labels = ['Low','Medium','High']
print('\n\nCLASSIFICATION REPORT FOR RANDOM FOREST\n')
print(classification_report(y_test,rfc_predictions, target_names=target_labels))

#Confusion Matrix
print('\n\nCONFUSION MATRIX FOR RANDOM FOREST\n')
print(metrics.confusion_matrix(y_test, rfc_predictions))

# 2. Support Vector Machine
from sklearn.svm import SVC

X = X_final
Y = y_final

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

svc = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

#Fitting the model
svc.fit(X_train,y_train)

#Making predictions
svc_predictions = svc.predict(X_test)

#Get results with metrics.accuracy_score or rfc.score(X_test, y_test)
svc_accu_score = round(metrics.accuracy_score(y_test, svc_predictions),4)
print('\n\n')
print('ACCURACY FOR SUPPORT VECTOR MACHINE :{}'.format(svc_accu_score))

#Classification Report
print('\n\nCLASSIFICATION REPORT FOR SUPPORT VECTOR MACHINE\n')
print(classification_report(y_test,svc_predictions, target_names=target_labels))

#Confusion Matrix
print('\n\nCONFUSION MATRIX FOR SUPPORT VECTOR MACHINE\n')
print(metrics.confusion_matrix(y_test, svc_predictions))

# 3. K-Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier

X = X_final
Y = y_final

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=9, p=2,
                     weights='uniform')

#Fitting the model
knn.fit(X_train,y_train)

#Making predictions
knn_predictions = knn.predict(X_test)

#Get results with metrics.accuracy_score or rfc.score(X_test, y_test)
knn_accu_score = round(metrics.accuracy_score(y_test, knn_predictions),4)
print('\n\n')
print('ACCURACY FOR K-NEAREST NEIGHBOR :{}'.format(knn_accu_score))

#Classification Report
print('\n\nCLASSIFICATION REPORT FOR K-NEAREST NEIGHBOR\n')
print(classification_report(y_test,knn_predictions, target_names=target_labels))

#Confusion Matrix
print('\n\nCONFUSION MATRIX FOR K-NEAREST NEIGHBOR\n')
print(metrics.confusion_matrix(y_test, knn_predictions))

