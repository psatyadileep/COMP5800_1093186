#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 
# 
# ---
# 
# **Author - Satya Dileep Penmetsa**
# 

# ### Import the required packages

# In[1]:


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import warnings
from pycm import *

# Disable warnings emitted by warnings.warn calls from different packages
# matplotlib can show a warning with tight_layout that can be safely ignored

warnings.filterwarnings('ignore')

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

# Scipy
from scipy import interpolate
from scipy import spatial
from scipy import stats
from scipy.cluster import hierarchy
# Others
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import pickle
from math import modf

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn import metrics
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from vecstack import stacking


sns.set(style='white', context='notebook', palette='deep')
from sklearn.model_selection import train_test_split, KFold, cross_validate

#Models
import warnings
warnings.filterwarnings("ignore")


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.inspection import plot_partial_dependence

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  
from sklearn.exceptions import ConvergenceWarning

pd.options.display.float_format = '{:.5f}'.format
import numpy as np
import matplotlib.pyplot as plt
plt.style.context('seaborn-talk')
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': '16',
          'figure.figsize': (15, 5),
         'axes.labelsize': '20',
         'axes.titlesize':'30',
         'xtick.labelsize':'18',
         'ytick.labelsize':'18'}
plt.rcParams.update(params)

plt.rcParams['text.color'] = '#A04000'
plt.rcParams['xtick.color'] = '#283747'
plt.rcParams['ytick.color'] = '#808000'
plt.rcParams['axes.labelcolor'] = '#283747'

# Seaborn style (figures)
sns.set(context='notebook', style='whitegrid')
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})


# # <a id="3">Read the data</a>

# In[2]:


data_df = pd.read_csv("data/creditcard.csv")


# In[3]:


print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])


# ## <a id="41">Explore the data</a>
# 
# We start by looking to the data features (first 5 rows).

# In[4]:


data_df.head()


# Let's look into more details to the data.

# In[5]:


data_df.describe()


# In[6]:


from pandas_profiling import ProfileReport
profile = ProfileReport(data_df, title="Pandas Profiling Report")
profile


# ## <a id="42">Check missing data</a>  
# 
# Let's check if there is any missing data.

# In[6]:


total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# There is no missing data in the entire dataset.

# ## <a id="43">Data unbalance</a>

# Let's check data unbalance with respect with *target* value, i.e. **Class**.

# In[7]:


temp = data_df["Class"].value_counts()
df = pd.DataFrame({'Class': temp.index,'values': temp.values})

trace = go.Bar(
    x = df['Class'],y = df['values'],
    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
    marker=dict(color="Blue"),
    text=df['values']
)
data = [trace]
layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
          xaxis = dict(title = 'Class', showticklabels=True), 
          yaxis = dict(title = 'Number of transactions'),
          hovermode = 'closest',width=1000
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='class')


# Only **492** (or **0.172%**) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable **Class**.

# # <a id="5">Data exploration</a>

# ## Transactions in time

# In[8]:


class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# Fraudulent transactions have a distribution more even than valid transactions - are equaly distributed in time, including the low real transaction times, during night in Europe timezone.

# ## Transactions amount

# In[9]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(17,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=False)
plt.show();


# In[10]:


tmp = data_df[['Amount','Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()


# In[11]:


class_1.describe()


# The real transaction have a larger mean value, larger Q1, smaller Q3 and Q4 and larger outliers; fraudulent transactions have a smaller Q1 and mean, larger Q4 and smaller outliers.
# 
# Let's plot the fraudulent transactions (amount) against time. The time is shown is seconds from the start of the time period (totaly 48h, over 2 days).

# In[12]:


fraud = data_df.loc[data_df['Class'] == 1]

trace = go.Scatter(
    x = fraud['Time'],y = fraud['Amount'],
    name="Amount",
     marker=dict(
                color='rgb(238,23,11)',
                line=dict(
                    color='red',
                    width=1),
                opacity=0.5,
            ),
    text= fraud['Amount'],
    mode = "markers"
)
data = [trace]
layout = dict(title = 'Amount of fraudulent transactions',
          xaxis = dict(title = 'Time [s]', showticklabels=True), 
          yaxis = dict(title = 'Amount'),
          hovermode='closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='fraud-amount')


# ## Features correlation
# 
# How does correlation help in feature selection? Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable. So, when two features have high correlation, we can drop one of the two features.

# In[13]:


plt.figure(figsize = (17,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data_df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()


# As expected, there is no notable correlation between features **V1**-**V28**. 

# ## Features density plot
# 
# The peaks of a Density Plot help display where values are concentrated over the interval. An advantage of Density Plots over Histograms is that they're better at determining the distribution shape because they're not affected by the number of bins. Normal distribution curves are an example of density plots.

# In[14]:


var = data_df.columns.values

i = 0
t0 = data_df.loc[data_df['Class'] == 0]
t1 = data_df.loc[data_df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# For some of the features we can observe a good selectivity in terms of distribution for the two values of **Class**: **V4**, **V11** have clearly separated distributions for **Class** values 0 and 1, **V12**, **V14**, **V18** are partially separated, **V1**, **V2**, **V3**, **V10** have a quite distinct profile, whilst **V25**, **V26**, **V28** have similar profiles for the two values of **Class**.  
# 
# In general, with just few exceptions (**Time** and **Amount**), the features distribution for legitimate transactions (values of **Class = 0**)  is centered around 0, sometime with a long queue at one of the extremities. In the same time, the fraudulent transactions (values of **Class = 1**) have a skewed (asymmetric) distribution.

# #### Data is IMBALANCED, Apply SMOTE

# In[6]:


data_df = data_df[data_df['Class']!='NA']
vals = data_df['Class'].value_counts()
print(vals)


# In[7]:



# Plotting attrition of employees
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14,6))

ax1 = data_df['Class'].value_counts().plot.pie(ax = ax1, shadow=True);
ax1.set(title = 'Number of Transactions by class')

ax2 = data_df['Class'].value_counts().plot(kind="barh" ,ax =ax2)
ax2.set(title = 'Number of Transactions by class')
plt.show()


# # Normalize attributes in dataframes

# In[8]:


final_df = data_df.copy()
final_df.head()


# In[9]:


final_df.drop(['Time'] ,axis=1, inplace=True)


# In[10]:


final_df.shape


# In[11]:


final_df.head()


# ## Over Sampling 
# 
# 
# 
# Synthetic Minority Oversampling Technique
# A problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary.
# 
# One way to solve this problem is to oversample the examples in the minority class. This can be achieved by simply duplicating examples from the minority class in the training dataset prior to fitting a model. This can balance the class distribution but does not provide any additional information to the model.
# 
# An improvement on duplicating examples from the minority class is to synthesize new examples from the minority class. This is a type of data augmentation for tabular data and can be very effective.
# 
# Perhaps the most widely used approach to synthesizing new examples is called the Synthetic Minority Oversampling TEchnique, or SMOTE for short. This technique was described by Nitesh Chawla, et al. in their 2002 paper named for the technique titled - SMOTE: Synthetic Minority Over-sampling Technique.
# 
# SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
# 
# Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.

# In[12]:



#Over-sampling: SMOTE
#SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, 
#based on those that already exist. It works randomly picking a point from the minority class and computing 
#the k-nearest neighbors for this point.The synthetic points are added between the chosen point and its neighbors.
#We'll use ratio='minority' to resample the minority class.
smote = SMOTE('minority')
finalDF = final_df.copy()
X = finalDF.drop(['Class'], axis=1)
Y = finalDF['Class']
X_sm, Y_sm = smote.fit_sample(X, Y)
print(X_sm.shape, Y_sm.shape)


# In[13]:


unique_elements, counts_elements = np.unique(Y_sm, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


# In[14]:


# Plotting attrition of employees
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14,6))

ax1 = pd.value_counts(Y_sm).plot(kind = 'pie', ax = ax1, shadow=True);
ax1.set(title = 'Number of Transactions by class')

ax2 = pd.value_counts(Y_sm).plot(kind="barh" ,ax =ax2)
ax2.set(title = 'Number of Transactions by class')
plt.show()


# ### Split into train and test

# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X_sm, Y_sm, test_size=0.30, shuffle=True)
print('Train data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))


# ## Our Modelling Approach

# <img src = "https://scikit-learn.org/stable/_images/grid_search_workflow.png">

# ## Simple modeling
# #### Cross validate models
# 
# I compared 10 popular classifiers and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
# 
# 
# - Random Forest
# - AdaBoost
# - Extra Trees
# - Gradient Boosting
# - Extreme Boosting

# ### Using Cross Validation for simpler models and their Comparison

# <img src = 'https://scikit-learn.org/stable/_images/grid_search_cross_validation.png'>

# ## Metrics
# 
# 
# **F1Score** is a metric to evaluate predictors performance using the formula
# 
# > F1 = 2 * (precision * recall) / (precision + recall)
# 
# where
# 
#  - recall = TP/(TP+FN) and precision = TP/(TP+FP)
# 
# When we have a multiclass setting, the average parameter in the f1_score function needs to be one of these:
# 
# - 'weighted'
# - 'micro'
# - 'macro'
# 
# The first one, 'weighted' calculates de F1 score for each class independently but when it adds them together uses a weight that depends on the number of true labels of each class: therefore favouring the majority class.
# 
# 'micro' uses the global number of TP, FN, FP and calculates the F1 directly: no favouring any class in particular.
# 
# Finally, 'macro' calculates the F1 separated by class but not using weights for the aggregation: which resuls in a bigger penalisation when your model does not perform well with the minority classes.
# 
# The one to use depends on what you want to achieve. If you are worried with class imbalance I would suggest using 'macro'. However, it might be also worthwile implementing some of the techniques available to taclke imbalance problems such as downsampling the majority class, upsampling the minority, SMOTE, etc.

# In[26]:


# Modeling step Test differents algorithms 
# Machine Learning Algorithm (MLA) Selection and Initialization

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=3)
random_state = 2
classifiers = []
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(XGBClassifier(random_state=random_state))
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, verbose = 1, n_jobs = -1, scoring = "f1", cv = kfold ))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())


# In[27]:


cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["RandomForest", "AdaBoost",
"ExtraTrees","GradientBoosting","XGBoost"]})

plt.style.use('bmh')
plt.figure(figsize=(17,8))
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy Score")
g = g.set_title("Cross validation scores")


# In[28]:


cv_res


# ### Looking at the Confusion Matrix

# In[30]:



f,ax=plt.subplots(3,2,figsize=(17,12))

y_pred = cross_val_predict(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state)),X_train,Y_train,cv=5)
sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('AdaBoost Classifier')

y_pred = cross_val_predict(RandomForestClassifier(random_state=random_state),X_train,Y_train,cv=5)
sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Random Forest Classifier')

y_pred = cross_val_predict(ExtraTreesClassifier(random_state=random_state),X_train,Y_train,cv=5)
sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Extra Trees Classifier')

y_pred = cross_val_predict(GradientBoostingClassifier(random_state=random_state),X_train,Y_train,cv=5)
sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Gradient Boosting Classifier')

y_pred = cross_val_predict(XGBClassifier(random_state = random_state),X_train,Y_train,cv=5)
sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('XG Boost Classifier')
                                                 
plt.subplots_adjust(hspace=0.4,wspace=0.5)
plt.show()


# ### Voting Classifier
# It is the simplest way of combining predictions from many different simple machine learning models. It gives an average prediction result based on the prediction of all the submodels. The submodels or the basemodels are all of diiferent types.

# In[32]:



ensemble = VotingClassifier(estimators=[('XGBoost', XGBClassifier(random_state = random_state)),
                                              ('Adaboost', AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1)),
                                              ('RF', RandomForestClassifier(random_state=random_state)),
                                              ('ET', ExtraTreesClassifier(random_state=random_state)),
                                              ('GBM', GradientBoostingClassifier(random_state=random_state))], 
                                               voting='hard').fit(X_train,Y_train)
cross=cross_val_score(ensemble,X_train,Y_train, cv = 5 ,scoring = "accuracy")
print('The cross validated score is',cross.mean())
print('The Accuracy score for ensembled model on test data is:',ensemble.score(X_test,Y_test))

y_pred_ensemble = cross_val_predict(ensemble,X_test,Y_test,cv=5)
plt.title('Ensemble Voting Classifier ')
sns.heatmap(confusion_matrix(Y_test,y_pred_ensemble),annot=True,fmt='2.0f')
print('\n clasification report:\n', classification_report(Y_test,y_pred_ensemble))
plt.show()


# # Comparison Table of ML based Models - Train Test Split

# In[17]:


from sklearn.metrics import cohen_kappa_score
Y_sm = Y_sm.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, test_size=0.30, shuffle=True)


train_accuracy = []
test_accuracy = []
precision = []
recall = []
f1 = []
cohen_kappa = []
models = ["RandomForest", "AdaBoost", "ExtraTrees","GradientBoosting","XGboost"]

random_state = 2
classifiers = []
classifiers.append(RandomForestClassifier(random_state=random_state, max_depth = 10, max_features = 'sqrt', n_estimators=  300))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.5))
classifiers.append(ExtraTreesClassifier(random_state=random_state, criterion ='entropy', max_features = 'sqrt', min_samples_leaf = 20, min_samples_split = 15))
classifiers.append(GradientBoostingClassifier(random_state=random_state, learning_rate = 0.2, max_depth = 10, n_estimators = 200))
classifiers.append(XGBClassifier(random_state = random_state))



for classifier,model in zip(classifiers, models):
    print('='*len(model))
    print(model)
    print('='*len(model))
    classifier.fit(X_train, y_train)
    trainprediction = classifier.predict(X_train)
    prediction = classifier.predict(X_test)
    trainaccuracy = accuracy_score(y_train, trainprediction)
    testaccuracy = accuracy_score(y_test, prediction)
    train_accuracy.append(trainaccuracy)
    test_accuracy.append(testaccuracy)
    precision.append(precision_score(y_test, prediction, average='macro'))
    recall.append(recall_score(y_test, prediction, average='macro'))
    cohen_kappa.append(cohen_kappa_score(y_test, prediction))
    f1.append(f1_score(y_test, prediction, average='macro'))
    print('\n clasification report:\n', classification_report(y_test,prediction))
    print('\n confussion matrix:\n',confusion_matrix(y_test, prediction))
    print('\n')
    
scoreDF = pd.DataFrame({'Model' : models})
scoreDF['Train Accuracy'] = train_accuracy
scoreDF['Test Accuracy'] = test_accuracy
scoreDF['Precision'] =  precision
scoreDF['Recall'] =  recall
scoreDF['F1 Score'] = f1 
scoreDF['Cohen Kappa Score'] = cohen_kappa


scoreDF.set_index("Model")


# In[18]:


scoreDF


# In[20]:


plt.style.use('fivethirtyeight')
def subcategorybar(X, vals, width=0.8):
    cols = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall','F1 Score', 'Cohen Kappa Score']
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], width=width/float(n), align="edge")   
        
        
    plt.xticks(_X, X)
    
plt.figure(figsize = (17,6))
subcategorybar(models, [scoreDF['Train Accuracy'], scoreDF['Test Accuracy'], scoreDF['Precision'], scoreDF['Recall'], scoreDF['F1 Score'], scoreDF['Cohen Kappa Score']])
plt.ylim(0.98, 1.0)
cols = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall','F1 Score', 'Cohen Kappa Score']
plt.legend(cols)
plt.xlabel('Model')
plt.title("Comparison of Models")
plt.show()


# # Ensemble on Simple base Models

# A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models.
# 
# It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble.
# 
# A voting ensemble works by combining the predictions from multiple models. It can be used for classification or regression. In the case of regression, this involves calculating the average of the predictions from the models. In the case of classification, the predictions for each label are summed and the label with the majority vote is predicted.
# 
# **Hard Voting** - Predict the class with the largest sum of votes from models
# 
# 
# **Soft Voting** - Predict the class with the largest summed probability from models.

# ### Simple Averaging

# In[16]:


random_state = 123
LR = LogisticRegression(random_state = random_state, n_jobs=-1)
NB = GaussianNB()
SVM = SVC(random_state=random_state)
KNN = KNeighborsClassifier(n_jobs=-1)
DT = DecisionTreeClassifier(random_state=random_state)
                                              
LR.fit(X_train, Y_train)
NB.fit(X_train, Y_train)
SVM.fit(X_train, Y_train)
KNN.fit(X_train, Y_train)
DT.fit(X_train, Y_train)

LR_pred = LR.predict(X_test)
NB_pred = NB.predict(X_test)
SVM_pred = SVM.predict(X_test)
KNN_pred = KNN.predict(X_test)
DT_pred = DT.predict(X_test)


# In[18]:


averaged_preds = (LR_pred + NB_pred + SVM_pred + KNN_pred + DT_pred)//5
acc = accuracy_score(Y_test, averaged_preds)
print('\n Accuracy Score:\n', np.round(acc, 3))
                            
print('\n clasification report:\n', classification_report(Y_test,averaged_preds))
print('\n confussion matrix:\n', metrics.confusion_matrix(Y_test, averaged_preds))
print('\n Cohen Kappa Score:\n', metrics.cohen_kappa_score(Y_test, averaged_preds))

print('\n')


# - In hard voting (also known as majority voting), every individual classifier votes for a class, and the majority wins. In statistical terms, the predicted target label of the ensemble is the mode of the distribution of individually predicted labels.
# 
# - In soft voting, every individual classifier provides a probability value that a specific data point belongs to a particular target class. The predictions are weighted by the classifier's importance and summed up. Then the target label with the greatest sum of weighted probabilities wins the vote.

# <img src = 'https://www.dropbox.com/s/59cyeldqfv2agc7/Screenshot%20%2832%29.png?dl=1'>

# ### Weighted Averaging

# In[17]:


ensemble = VotingClassifier(estimators=[('Logistic Regression', LogisticRegression(random_state = random_state)),
                                              ('Naive Bayes', GaussianNB()),
                                              ('SVM', SVC(random_state=random_state)),
                                              ('KNN', KNeighborsClassifier()),
                                              ('Decision Tree', DecisionTreeClassifier(random_state=random_state))], 
                                               voting='hard').fit(X_train,Y_train)

y_pred_ensemble = ensemble.predict(X_test)

print('\n Accuracy Score:\n', np.round(accuracy_score(Y_test, y_pred_ensemble), 3))
sns.heatmap(confusion_matrix(Y_test,y_pred_ensemble),annot=True,fmt='2.0f')
print('\n clasification report:\n', classification_report(Y_test,y_pred_ensemble))
print('\n Cohen Kappa Score:\n', metrics.cohen_kappa_score(Y_test, y_pred_ensemble))

plt.show()


# ### Max Voting

# In[19]:


ensemble = VotingClassifier(estimators=[('Logistic Regression', LogisticRegression(random_state = random_state)),
                                              ('Naive Bayes', GaussianNB()),
                                              ('SVM', SVC(random_state=random_state, probability = True)),
                                              ('KNN', KNeighborsClassifier()),
                                              ('Decision Tree', DecisionTreeClassifier(random_state=random_state))], 
                                               voting='soft').fit(X_train,Y_train)

y_pred_ensemble = ensemble.predict(X_test)
print('\n Accuracy Score:\n', np.round(accuracy_score(Y_test, y_pred_ensemble), 3))
sns.heatmap(confusion_matrix(Y_test,y_pred_ensemble),annot=True,fmt='2.0f')
print('\n clasification report:\n', classification_report(Y_test,y_pred_ensemble))
print('\n Cohen Kappa Score:\n', metrics.cohen_kappa_score(Y_test, y_pred_ensemble))
plt.show()


# # Ensmble of multiple Models with a Generalised Model

# In[41]:


models = [

    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    SGDClassifier(max_iter=100, tol=1e-4),

    DecisionTreeClassifier(),
    RandomForestClassifier(criterion='entropy',n_estimators=100),
    BaggingClassifier(),
    xgb.XGBClassifier(n_estimators= 100,objective = 'binary:logistic'),
    
    
    AdaBoostClassifier(),
    GradientBoostingClassifier(n_estimators=100,max_features='sqrt'),
    ExtraTreesClassifier(n_estimators= 100)
]

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

Y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
Y_test.replace([np.inf, -np.inf], np.nan, inplace=True)


# ## Stacked Ensemble <a id='stack-ensemble'></a>
# 
# Stacking, also called Super Learning or Stacked Regression, is a class of algorithms that involves training a second-level **“metalearner”** to find the optimal combination of the base learners. Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together.
# Reference: [H2O](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
# 
# ### Stacking Algorithm
# 
# **Step 1. Set up the ensemble.**<br>
#  a. Specify a list of L base algorithms (with a specific set of model parameters)..<br>
#  b. Specify a metalearning algorithm..<br>
#  
# **Step 2. Train the ensemble.**<br>
#  a. Train each of the L base algorithms on the training set..<br>
#  b. Perform k-fold cross-validation on each of these learners and collect the cross-validated predicted values from each of the L algorithms..<br>
#  c. The N cross-validated predicted values from each of the L algorithms can be combined to form a new N x L matrix. This matrix, along wtih the original response vector, is called the “level-one” data. (N = number of rows in the training set.).<br>
#  d. Train the metalearning algorithm on the level-one data. The “ensemble model” consists of the L base learning models and the metalearning model, which can then be used to generate predictions on a test set.<br>
#  
# **Step 3. Predict on new data.**<br>
#  a. To generate ensemble predictions, first generate predictions from the base learners.<br>
#  b. Feed those predictions into the metalearner to generate the ensemble prediction.<br>
# 
# ![](https://i.ibb.co/3cRfkxK/stacked-ensemble.png)

# Now, as per performance of different baseline models on cross validation accuracy we will be selecting best performing models for level 0 of **stacked ensemble** so that their ensemble will produce higher performance in comparison to individual machine learning model.

# In[30]:


X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

Y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
Y_test.replace([np.inf, -np.inf], np.nan, inplace=True)

S_train, S_test = stacking(models,                   
                           X_train, Y_train, X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=3, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)


# In[33]:


# initializing generalizer model i.e., MLP classifier in our case
model = RandomForestClassifier()
    
model = model.fit(S_train, Y_train)
y_pred = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(Y_test, y_pred))
sns.heatmap(confusion_matrix(Y_test,y_pred),annot=True,fmt='2.0f')
print('\n clasification report:\n', classification_report(Y_test,y_pred))
plt.show()


# ## Model Evaluation  <a id='model-eval'></a>
# 
#  In this step we will first define which evaluation metrics we will use to evaluate our model. The most important evaluation metric for this problem domain is **sensitivity, specificity, Precision, F1-measure, Geometric mean and mathew correlation coefficient and finally ROC AUC curve**
#  
#  
# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. The confusion matrix itself is relatively simple to understand, but the related terminology can be confusing.
# 
# Example confusion matrix for a binary classifier
# 
# <img src = 'https://www.dataschool.io/content/images/2015/01/confusion_matrix_simple2.png'>
# 
# What can we learn from this matrix?
# 
# - There are two possible predicted classes: "yes" and "no". If we were predicting the presence of a disease, for example, "yes" would mean they have the disease, and "no" would mean they don't have the disease.
# - The classifier made a total of 165 predictions (e.g., 165 patients were being tested for the presence of that disease).
# - Out of those 165 cases, the classifier predicted "yes" 110 times, and "no" 55 times.
# - In reality, 105 patients in the sample have the disease, and 60 patients do not.
# 
# Let's now define the most basic terms, which are whole numbers (not rates):
# 
# **true positives (TP):** These are cases in which we predicted yes (they have the disease), and they do have the disease.
# 
# **true negatives (TN):** We predicted no, and they don't have the disease.
# 
# **false positives (FP):** We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
# 
# **false negatives (FN):** We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
# 
# I've added these terms to the confusion matrix, and also added the row and column totals:
# 
# 
# This is a list of rates that are often computed from a confusion matrix for a binary classifier:
# 
# **Accuracy:** Overall, how often is the classifier correct?
# (TP+TN)/total = (100+50)/165 = 0.91
# 
# **Misclassification Rate:** Overall, how often is it wrong?
# (FP+FN)/total = (10+5)/165 = 0.09
# equivalent to 1 minus Accuracy
# also known as "Error Rate"
# 
# **True Positive Rate:** When it's actually yes, how often does it predict yes?
# TP/actual yes = 100/105 = 0.95
# also known as "Sensitivity" or "Recall"
# 
# **False Positive Rate:** When it's actually no, how often does it predict yes?
# FP/actual no = 10/60 = 0.17
# 
# **True Negative Rate:** When it's actually no, how often does it predict no?
# TN/actual no = 50/60 = 0.83
# equivalent to 1 minus False Positive Rate
# also known as "Specificity"
# 
# **Precision:** When it predicts yes, how often is it correct?
# TP/predicted yes = 100/110 = 0.91
# 
# **Prevalence:** How often does the yes condition actually occur in our sample?
# actual yes/total = 105/165 = 0.64
# A couple other terms are also worth mentioning:
# 
# **Null Error Rate:** This is how often you would be wrong if you always predicted the majority class. (In our example, the null error rate would be 60/165=0.36 because if you always predicted yes, you would only be wrong for the 60 "no" cases.) 
# 
# This can be a useful baseline metric to compare your classifier against. However, the best classifier for a particular application will sometimes have a higher error rate than the null error rate, as demonstrated by the Accuracy Paradox.
# 
# 
# **Cohen's Kappa:** This is essentially a measure of how well the classifier performed as compared to how well it would have performed simply by chance. In other words, a model will have a high Kappa score if there is a big difference between the accuracy and the null error rate. (More details about Cohen's Kappa.)
# 
# **F Score:** This is a weighted average of the true positive rate (recall) and precision. (More details about the F Score.)
# 
# **ROC Curve:** This is a commonly used graph that summarizes the performance of a classifier over all possible thresholds. It is generated by plotting the True Positive Rate (y-axis) against the False Positive Rate (x-axis) as you vary the threshold for assigning observations to a given class. (More details about ROC Curves.)
# 
# 
# 
# ### Sensitivity vs Specificity

# ![](https://i.ibb.co/d43FVfJ/Sensitivity-and-specificity-svg.png)
# 
# ### Mathew Correlation coefficient (MCC)
# 
# The Matthews correlation coefficient (MCC), instead, is a more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset.
# 
# ![](https://i.ibb.co/mH6MmG4/mcc.jpg)

# ### Log Loss
# Logarithmic loss  measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0. Log loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high log loss.
# 
# The graph below shows the range of possible log loss values given a true observation (isDog = 1). As the predicted probability approaches 1, log loss slowly decreases. As the predicted probability decreases, however, the log loss increases rapidly. Log loss penalizes both types of errors, but especially those predications that are confident and wrong!
# 
# ![](https://i.ibb.co/6BdDczW/log-loss.jpg)

# ### F1 Score
# 
#  F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701.
# 
# **F1 Score = 2*(Recall * Precision) / (Recall + Precision)**
# 
# 
# <img src = 'https://www.researchgate.net/publication/325567208/figure/tbl4/AS:668664739151911@1536433505975/Classification-performance-metrics-based-on-the-confusion-matrix.png'>

# In[28]:


from sklearn import metrics
CM=confusion_matrix(Y_test,y_pred)
sns.heatmap(CM, annot=True)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
specificity = TN/(TN+FP)
loss_log = metrics.log_loss(Y_test, y_pred)
acc= accuracy_score(Y_test, y_pred)
roc=metrics.roc_auc_score(Y_test, y_pred)
prec = metrics.precision_score(Y_test, y_pred)
rec = metrics.recall_score(Y_test, y_pred)
f1 = metrics.f1_score(Y_test, y_pred)
cohen_kappa = metrics.cohen_kappa_score(Y_test, y_pred)
mathew = metrics.matthews_corrcoef(Y_test, y_pred)

model_results =pd.DataFrame([['Stacked Classifier',acc, prec,rec,specificity, f1,roc, loss_log,mathew, cohen_kappa]],
               columns = ['Model', 'Accuracy','Precision', 'Sensitivity','Specificity', 'F1 Score','ROC','Log_Loss','mathew_corrcoef', 'Cohen Kappa'])

model_results


# ## Comparison with other Models

# In[52]:


models = [

    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    SGDClassifier(max_iter=100, tol=1e-4),

    DecisionTreeClassifier(),
    RandomForestClassifier(criterion='entropy',n_estimators=100),
    BaggingClassifier(),
    xgb.XGBClassifier(n_estimators= 100,objective = 'binary:logistic'),
    
    
    AdaBoostClassifier(),
    GradientBoostingClassifier(n_estimators=100,max_features='sqrt'),
    ExtraTreesClassifier(n_estimators= 100)
]


# In[69]:


SGD = SGDClassifier(max_iter=100, tol=0.0001, loss = 'log')
SGD.fit(X_train, Y_train)


# In[45]:


X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

Y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
Y_test.replace([np.inf, -np.inf], np.nan, inplace=True)


LR = LogisticRegression()
LDA = LinearDiscriminantAnalysis()
SGD = SGDClassifier(max_iter=100, tol=0.0001, loss = 'log')
DT = DecisionTreeClassifier()
RF = RandomForestClassifier(criterion='entropy')
BG = BaggingClassifier()
XGB = XGBClassifier()
ADA = AdaBoostClassifier()
GBM = GradientBoostingClassifier(max_features='sqrt')
ET = ExtraTreesClassifier()

LR.fit(X_train, Y_train)
LDA.fit(X_train, Y_train)
SGD.fit(X_train, Y_train)
DT.fit(X_train, Y_train)
RF.fit(X_train, Y_train)
BG.fit(X_train, Y_train)
XGB.fit(X_train, Y_train)
ADA.fit(X_train, Y_train)
GBM.fit(X_train, Y_train)
ET.fit(X_train, Y_train)


y_pred_LR = LR.predict(X_test) 
y_pred_LDA  = LDA.predict(X_test)
y_pred_SGD  = SGD.predict(X_test)
y_pred_DT = DT.predict(X_test)
y_pred_RF   = RF.predict(X_test)
y_pred_BG  = BG.predict(X_test)
y_pred_XGB = XGB.predict(X_test)
y_pred_ADA  = ADA.predict(X_test)
y_pred_GBM  = GBM.predict(X_test)
y_pred_ET  = ET.predict(X_test)


# In[58]:


data = {        'LR': y_pred_LR, 
                'LDA': y_pred_LDA, 
                'SGD': y_pred_SGD, 
                'DT': y_pred_DT,
                'RF': y_pred_RF, 
                'BG': y_pred_BG, 
                'XGB': y_pred_XGB,
                'Adaboost': y_pred_ADA, 
                'GBM': y_pred_GBM, 
                'ET': y_pred_ET}

models = pd.DataFrame(data) 


# In[33]:


for column in models:
    CM=confusion_matrix(Y_test,models[column])
    
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    specificity = TN/(TN+FP)
    loss_log = metrics.log_loss(Y_test, models[column])
    acc= accuracy_score(Y_test, models[column])
    roc=metrics.roc_auc_score(Y_test, models[column])
    prec = metrics.precision_score(Y_test, models[column])
    rec = metrics.recall_score(Y_test, models[column])
    f1 = metrics.f1_score(Y_test, models[column])
    
    mathew = metrics.matthews_corrcoef(Y_test, models[column])
    cohen_kappa = metrics.cohen_kappa_score(Y_test, models[column])
    results =pd.DataFrame([[column,acc, prec,rec,specificity, f1,roc, loss_log,mathew, cohen_kappa]],
               columns = ['Model', 'Accuracy','Precision', 'Sensitivity','Specificity', 'F1 Score','ROC','Log_Loss','mathew_corrcoef', 'Cohen Kappa'])
    model_results = model_results.append(results, ignore_index = True)

model_results


# In[35]:


model_resultsDF = model_results.copy()
model_resultsDF.to_csv('model_resultsDF.csv', index = False)


# ### Findings
# - AS we can see from above results, Stacked Ensemble Classifier is best performer as it has highest test accuracy of 0.906, sensitivity of 0.99994 and specificity of 1.0 and highest f1-score of 0.99994 and lowest Log Loss of 0.00223 and highest ROC value of 0.99994
# - Extra Tree, Xgboost and bagging Classifier classifier are second best having same performance measure in every aspect

# ### ROC AUC Curve

# The Receiver Operator Characteristic (ROC) curve is an evaluation metric for binary classification problems. It is a probability curve that plots the TPR against FPR at various threshold values and essentially separates the ‘signal’ from the ‘noise’. The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve.
# 
# The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.
# 
# 
# When AUC = 1, then the classifier is able to perfectly distinguish between all the Positive and the Negative class points correctly. If, however, the AUC had been 0, then the classifier would be predicting all Negatives as Positives, and all Positives as Negatives.
# 
# 
# When 0.5<AUC<1, there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.
# 
# 
# When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points. Meaning either the classifier is predicting random class or constant class for all the data points.
# 
# So, the higher the AUC value for a classifier, the better its ability to distinguish between positive and negative classes.

# In[71]:


def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

f, ax = plt.subplots(figsize=(12,8))

#roc_auc_plot(Y_test,model.predict_proba(S_test),label='Stacked Classifier ',l='-')
roc_auc_plot(Y_test,LR.predict_proba(X_test),label='Logsitic Regression Classifier ',l='-')
roc_auc_plot(Y_test,LDA.predict_proba(X_test),label='LDA Classifier ',l='-')
roc_auc_plot(Y_test,SGD.predict_proba(X_test),label='SGD Classifier',l='-')
roc_auc_plot(Y_test,DT.predict_proba(X_test),label='Decision TreeClassifier ',l='-')
roc_auc_plot(Y_test,RF.predict_proba(X_test),label='Random Forest Classifier ',l='-')
roc_auc_plot(Y_test,BG.predict_proba(X_test),label='Bagging Classifier ',l='-')
roc_auc_plot(Y_test,XGB.predict_proba(X_test),label='XGboost',l='-')
roc_auc_plot(Y_test,ADA.predict_proba(X_test),label='Adaooost Classifier ',l='-')
roc_auc_plot(Y_test,GBM.predict_proba(X_test),label='Gradient Boosting Machine Classifier ',l='-')
roc_auc_plot(Y_test,ET.predict_proba(X_test),label='Extra Trees Classifier ',l='-')

ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 
        )    
ax.legend(loc="lower right")    
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Receiver Operator Characteristic curves')
sns.despine()


# As we can see highest average area under the curve (AUC) of 0.950 is attained by Extra Tree Classifier

# ## Precision Recall curve

# Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.

# In[73]:


def precision_recall_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(Y_test,
                                                  y_proba[:,1])
    average_precision = average_precision_score(Y_test, y_proba[:,1],
                                                     average="micro")
    ax.plot(recall, precision, label='%s (average=%.3f)'%(label,average_precision),
            linestyle=l, linewidth=lw)

f, ax = plt.subplots(figsize=(14,10))
#precision_recall_plot(Y_test,model.predict_proba(S_test),label='Stacked Classifier ',l='-')
precision_recall_plot(Y_test,LR.predict_proba(X_test),label='Logsitic Regression Classifier ',l='-')
precision_recall_plot(Y_test,LDA.predict_proba(X_test),label='LDA Classifier ',l='-')
precision_recall_plot(Y_test,SGD.predict_proba(X_test),label='SGD Classifier',l='-')
precision_recall_plot(Y_test,DT.predict_proba(X_test),label='Decision TreeClassifier ',l='-')
precision_recall_plot(Y_test,RF.predict_proba(X_test),label='Random Forest Classifier ',l='-')
precision_recall_plot(Y_test,BG.predict_proba(X_test),label='Bagging Classifier ',l='-')
precision_recall_plot(Y_test,XGB.predict_proba(X_test),label='XGboost',l='-')
precision_recall_plot(Y_test,ADA.predict_proba(X_test),label='Adaooost Classifier ',l='-')
precision_recall_plot(Y_test,GBM.predict_proba(X_test),label='Gradient Boosting Machine Classifier ',l='-')
precision_recall_plot(Y_test,ET.predict_proba(X_test),label='Extra Trees Classifier ',l='-')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc="lower left")
ax.grid(True)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Precision-recall curves')
sns.despine()


# # Feature Selection 

# Here, we are using multiple features in different feature selection techniques to see which features are consider inportant by different techniques 

# In[75]:


num_feats=11

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X_sm, Y_sm,num_feats)
print(str(len(cor_feature)), 'selected features')


# In[76]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X_sm)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, Y_sm)
chi_support = chi_selector.get_support()
chi_feature = X_sm.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')


# In[78]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, Y_sm)
rfe_support = rfe_selector.get_support()
rfe_feature = X_sm.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')


# In[79]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2", solver='lbfgs'), max_features=num_feats)
embeded_lr_selector.fit(X_norm, Y_sm)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X_sm.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')


# In[80]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, criterion='entropy'), max_features=num_feats)
embeded_rf_selector.fit(X_sm, Y_sm)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X_sm.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# In[82]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
embeded_lgb_selector.fit(X_sm, Y_sm)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X_sm.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')


# In[83]:


# put all selection together
feature_name = X_sm.columns
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# ## Deep Learning based Ensembles
# 
# 
# Neural network models are nonlinear and have a high variance, which can be frustrating when preparing a final model for making predictions.
# 
# Ensemble learning combines the predictions from multiple neural network models to reduce the variance of predictions and reduce generalization error.
# 
# Techniques for ensemble learning can be grouped by the element that is varied, such as training data, the model, and how predictions are combined.
# 
# 
# 
# #1 Stacking: in stacking the output of the base-learners are taken as input for training a meta-learner, that learns how to best combine the base-learners predictions.
# 
# 
# <img src = 'https://miro.medium.com/max/963/1*1ArQEf8OFkxVOckdWi7mSA.png'>
# 
# Stacking combines multiple predictive models in order to generate a new combined model.
# Often times the stacking model will outperform each of the individual models due to its smoothing nature, offsetting deficiencies of individual models leading to a better prediction performance. Therefore, stacking works best when the base models are basically different.
# 
# 
# #2 Weighted Average Ensemble: this method weights the contribution of each ensemble member based on their performance on a hold-out validation dataset. Models with better contribution receive a higher weight.
# 
# 
# <img src = 'https://miro.medium.com/max/952/1*5CnIeN_BtByepM_4JWrdvQ.png'>
# 
# 
# Weighted Average Ensemble is all about weighting the predictions of each base-model generating a combined prediction.
# The main difference between both methods is that in stacking, the meta-learner takes every single output of the base-learners as a training instance, learning how to best map the base-learner decisions into an improved output. The meta-learner can be any classic known machine learning model. The weighted average ensemble on the other hand is just about optimizing weights that are used for weighting all outputs of the base-learner and taking the weighted average. There is no meta-learner here (besides the weights). Here the number of weights is equal to the number of existing base-learners.
# 
# 

# In[32]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam # I believe this is better optimizer for our case
from keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from keras.utils.vis_utils import plot_model
import scipy


# In[66]:


scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)


# In[80]:


def model_ANN1(input_shape=X_train.shape[1], num_classes=2):   
    model = Sequential()

    model.add(Dense(32, activation='tanh', input_dim=X_train.shape[1]))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.4))

    # Lets add softmax activated neurons as much as number of classes
    model.add(Dense(1, activation = "sigmoid"))
    # Compile the model with loss and metrics
    model.compile(optimizer =  Adam(learning_rate = 0.00001) , loss = "binary_crossentropy", metrics=["accuracy"])
    
    return model


# In[81]:


def model_ANN2(input_shape=X_train.shape[1], num_classes=2):   
    model = Sequential()

    model.add(Dense(128, activation='tanh',  input_dim=X_train.shape[1]))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))

    # Lets add softmax activated neurons as much as number of classes
    model.add(Dense(1, activation = "sigmoid"))
    # Compile the model with loss and metrics
    model.compile(optimizer =  Adam(learning_rate = 0.00001) , loss = "binary_crossentropy", metrics=["accuracy"])
    
    return model


# In[82]:


ANN_model1 = model_ANN1(input_shape=X_train.shape[1], num_classes=2)
ANN_model2 = model_ANN2(input_shape=X_train.shape[1], num_classes=2)


# # Stacking DL models

# In[83]:


model = []
model.append(ANN_model1)
model.append(ANN_model2)


# In[84]:


# Start multiple model training with the batch size
# Use Reduce LR on Plateau for reducing Learning Rate if there is no decrease at loss for 3 epochs
models = []
for i in range(len(model)):
    model[i].fit(X_train,Y_train, batch_size=128,
                                        epochs = 10,
                                        validation_data = (X_test,Y_test), 
                                        callbacks=[ReduceLROnPlateau(monitor='loss', patience=3, factor=0.1)], 
                                        verbose=2)
    models.append(model[i])


# In[24]:


ANN1_preds = (model[0].predict(X_test) > 0.5).astype("int32")
ANN2_preds = (model[1].predict(X_test) > 0.5).astype("int32")
preds = pd.DataFrame({"ANN1" : ANN1_preds.ravel(), "ANN2" : ANN2_preds.ravel()})

## Considering majority Voting

ANN_ensemble_predicted = preds.mode(axis=1)
final_preds = ANN_ensemble_predicted.iloc[:, 0].astype("int32")

from sklearn import metrics
CM=metrics.confusion_matrix(Y_test,final_preds)
sns.heatmap(CM, annot=True)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
specificity = TN/(TN+FP)
loss_log = metrics.log_loss(Y_test, final_preds)
acc= accuracy_score(Y_test, final_preds)
roc=metrics.roc_auc_score(Y_test, final_preds)
prec = metrics.precision_score(Y_test, final_preds)
rec = metrics.recall_score(Y_test, final_preds)
f1 = metrics.f1_score(Y_test, final_preds)

mathew = metrics.matthews_corrcoef(Y_test, final_preds)
cohen_kappa = metrics.cohen_kappa_score(Y_test, final_preds)

model_results =pd.DataFrame([['ANN Emsemble Classifier',acc, prec,rec,specificity, f1,roc, loss_log,mathew, cohen_kappa]],
               columns = ['Model', 'Accuracy','Precision', 'Sensitivity','Specificity', 'F1 Score','ROC','Log_Loss','mathew_corrcoef', 'Cohen Kappa'])

model_results


# ## Model Interpretation  <a id='model-inter'></a>

# ### SHAP Feature Importance
# The global mean(|Tree SHAP|) method applied to the heart disease prediction model. The x-axis is essentially the average magnitude change in model output when a feature is “hidden” from the model (for this model the output has log-odds units). “hidden” means integrating the variable out of the model. Since the impact of hiding a feature changes depending on what other features are also hidden, Shapley values are used to enforce consistency and accuracy.

# In[85]:


import shap 
explainer = shap.TreeExplainer(RF)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")


# ### SHAP Summary Plot

# In[86]:


shap.summary_plot(shap_values[1], X_test)


# The higher the SHAP value of a feature, the higher is the log odds of heart disease in this heart disease prediction model. Every patient in the dataset is run through the model and a dot is created for each feature attribution value, so one patient gets one dot on each feature’s line. Dot’s are colored by the feature’s value for that patient and pile up vertically to show density. In above plot we see that **st_slope_upsloping** is the most important risk factor for heart disease patients. The lower values of **st_slope_upsloping** leads to heart disease, whereas its higher values decreases the chances of heart disease. Higher values of **exercise_induced_angina** increases the risk of heart disease whereas its lower values decreases the chances of heart disease.

# ## Conclusion  <a id='data-conc'></a>
# 
# - As we have seen, stacked ensemble of power machine learning algorithms resulted in higher performance than any individual machine learning model.
# - We have also interpreted second best performing algo i.e., random forest algorithm
# - The top 5 most contribution features are -- 
# 

# In[ ]:





# In[ ]:




