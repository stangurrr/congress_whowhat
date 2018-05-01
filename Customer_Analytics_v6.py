"""
Customer Analytics
@author: steve

- Feature selection
    - Features <5% of rows
    - PCA
    - http://www.quora.com/How-do-I-perform-feature-selection
- SQLAlchemy - connect to SQL engine 
- GridSearch
- Ensemble?
- Flask, React


Help:
- Seaborn - cluster map

To-do:
- Regularization
- K-Means
- Linear Regression - customer lifetime

- Export Lifetime, Reviews Per Mo, Cumulative Reviews

"""
''' DATA IMPORT '''
# python libraries
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# data import
df_accounts = pd.read_csv('Cust_Analytics_v5.csv')
df_dental = df_accounts[(df_accounts.Industry == 'Dental')]
df_dental.isnull().sum()


df_accounts.shape
df_dental.shape

df_dental.groupby('Rating').count()
df_dental.groupby('Rating')['AccountID', 'Sub_Start_Date'].count() #issue w/ cancelled accounts not having cancellation date
df_dental.groupby('Rating')['AccountID', 'Cancellation_Date'].count() #issue w/ cancelled accounts not having cancellation date
df_dental.groupby('Rating')['AccountID', 'Cancellation_Requested_Date'].count() #issue w/ cancelled accounts not having cancellation date

''' DATA EXPLORATION '''
# data exploration
df_dental.head()
df_dental.describe()
df_dental.columns
df_dental['D3One Mgmt System (Most Recent)'].unique()
'''Q: Do we know the integration type by M/S? '''

sns.pairplot(df_accounts)
df_accounts.corr()
sns.heatmap(df_dental.corr())


''' DATA PREPARATION '''
# drop cancelled accounts w/o cancellation request dates
df_dental['Sub_Start_Date'] = pd.to_datetime(df_dental['Sub_Start_Date'])
df_dental['Cancellation_Date'] = pd.to_datetime(df_dental['Cancellation_Date'])
df_dental['Cancellation_Requested_Date'] = pd.to_datetime(df_dental['Cancellation_Requested_Date'])
df_dental['D3 Last Login Date to Portal'] = pd.to_datetime(df_dental['D3 Last Login Date to Portal'])
present = datetime(2015, 7, 20)
df_dental['Cancellation_Requested_Date'] = df_dental['Cancellation_Requested_Date'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '')
df_dental = df_dental.drop(df_dental[(df_dental.Rating =='Cancelled') & (df_dental.Cancellation_Requested_Date == '')].index)
df_dental['Cancellation_Requested_Date'] = pd.to_datetime(df_dental['Cancellation_Requested_Date'])

# data preparation for null fields
df_dental.groupby('Rating').count()
df_dental.groupby('Rating')['AccountID', 'Sub_Start_Date'].count() #issue w/ cancelled accounts not having cancellation date
df_dental.groupby('Rating')['AccountID', 'Cancellation_Date'].count() #issue w/ cancelled accounts not having cancellation date
df_dental.groupby('Rating')['AccountID', 'Cancellation_Requested_Date'].count() #issue w/ cancelled accounts not having cancellation date
'''Q: How are sub/cxl dates being managed in analysis now? '''


''' CUSTOMER LIFETIME CALCULATION '''
# customer subscription lifetime
def Biz_CustLife_calc(row):
    if row.Rating == 'Under Contract':
        return present - row.Sub_Start_Date
    elif row.Rating == 'Cancelled':
        return row.Cancellation_Requested_Date - row.Sub_Start_Date

df_dental['Biz_CustLife'] = df_dental.apply(Biz_CustLife_calc, axis=1)
df_dental['Biz_CustLife'].isnull().sum()

df_dental['Biz_CustLife'] = df_dental['Biz_CustLife'] / np.timedelta64(1,'s')
df_dental['Biz_CustLife'] = df_dental['Biz_CustLife'] / 60 / 60 / 24 / 365
df_dental = df_dental.drop(df_dental[df_dental.Biz_CustLife <= 0].index)

df_dental.shape
df_dental['Biz_CustLife'].describe()


''' FEATURE ENGINEERING '''
df_dental.columns
df_dental.rename(columns={'D3One Mgmt System (Most Recent)': 'Mgmt_Sys', 
                          'D3 Google Places Account': 'Google+',
                          'D3 Number of Customers': '#Customers', 
                          'D3 Number of Emails': '#Emails'
                          }, inplace=True)

# Appointments / Customer Life
df_dental['D3 Number of Appointments'].fillna(0, inplace=True)
df_dental['Usage_#Appts_CumMoAvg'] = [float(a)/b for a,b in zip(df_dental['D3 Number of Appointments'], df_dental['Biz_CustLife'])]
df_dental['Usage_#Appts_CumMoAvg'].describe()

# Campaigns Sent / Customer Life
df_dental['Usage_#Camp_CumMoAvg'] = [float(a)/b for a,b in zip(df_dental['D3 Number of Campaigns Sent'], df_dental['Biz_CustLife'])]
df_dental['Usage_#Camp_CumMoAvg'].describe()

# DF Appts Requested / Customer Life
df_dental['D3 Number of Appointments Requested'].fillna(0, inplace=True)
df_dental['Usage_#DFApptReq_CumMoAvg'] = [float(a)/b for a,b in zip(df_dental['D3 Number of Appointments Requested'], df_dental['Biz_CustLife'])]
df_dental['Usage_#DFApptReq_CumMoAvg'].describe()

# Public Reviews / Customer Life
df_dental['Usage_#PublicReviews_CumMoAvg'] = [float(a)/b for a,b in zip(df_dental['D3 Public Reviews'], df_dental['Biz_CustLife'])]
df_dental['Usage_#PublicReviews_CumMoAvg'].describe()

# % of Total Appointments Confirmed
df_dental['Usage_%ApptConfirmRate_Cum'] = [float(a)/b for a,b in zip(df_dental['D3 Confirmed Appointments'],df_dental['D3 Number of Appointments'])]
df_dental['Usage_%ApptConfirmRate_Cum'].fillna(0, inplace=True)
df_dental['Usage_%ApptConfirmRate_Cum'].describe()

# More data clean-up!
df_dental.columns
df_dental.Rating = np.where(df_dental.Rating == 'Under Contract', 1, 0)
df_dental['StackedDiscount_Ind'] = np.where(df_dental['StackedDiscount_Ind'] == 'Stacked Discount', 1, 0)

df_dental['D3 Web Alias'] = np.where(df_dental['D3 Web Alias'] > 0, 1, 0)
df_dental['Google+'] = np.where(df_dental['Google+'] > 0, 1, 0)
df_dental['D3 Yelp Account'] = np.where(df_dental['D3 Yelp Account'] > 0, 1, 0)
df_dental['D3 Facebook Account'] = np.where(df_dental['D3 Facebook Account'] > 0, 1, 0)
df_dental['Google+'] = np.where(df_dental['Google+'] > 0, 1, 0)
df_dental['D3 OOB Campaigns Enabled'].fillna(0, inplace=True)
df_dental['D3 Total Emails Sent'].fillna(0, inplace=True)
df_dental['D3 Custom Campaigns Emails Sent'].fillna(0, inplace=True)
df_dental['D3 Public Reviews'].fillna(0, inplace=True)
df_dental['Usage_#Camp_CumMoAvg'].fillna(0, inplace=True)
df_dental['Usage_#PublicReviews_CumMoAvg'].fillna(0, inplace=True)
df_dental['#Customers'].fillna(0, inplace=True)
df_dental['#Emails'].fillna(0, inplace=True)

''' CUSTOMER COHORT CREATION '''
def get_time_bins(val):
    our_quarter = {
        1: 2,
        2: 3,
        3: 3,
        4: 3,
        5: 4,
        6: 4,
        7: 4,
        8: 1,
        9: 1,
        10: 1,
        11: 2,
        12: 2,
    }
    return str(val.year) + '-Q' + str(our_quarter[val.month])
df_dental['cohort'] = df_dental.Sub_Start_Date.map(get_time_bins)
FY14Q3_cohort = df_dental[df_dental['cohort'] == '2014-Q3']


df_dental[['Sub_Start_Date', 'cohort']]    
df_dental.groupby('cohort')['Rating'].count()

# Accounts Per Cohort Chart
df_dental.cohort.value_counts().sort_index().plot()
df_dental.boxplot(column='Biz_CustLife', by='Rating')

# Cohort Retention Chart

FY14Q3_cohort.groupby('Rating')['AccountID', 'Cancellation_Requested_Date'].count() #issue w/ cancelled accounts not having cancellation date

FY14Q3_cohort.groupby('Rating').Biz_CustLife.describe()
FY14Q3_cohort.shape[0]

FY14Q3_cohort.to_csv('FY14Q3_cohort.csv') #check data

# Data Exploration
FY14Q3_cohort.boxplot(column='Biz_CustLife', by='Rating')
FY14Q3_cohort.boxplot(column='Usage_#Appts_CumMoAvg', by='Rating', showfliers=False)
FY14Q3_cohort.boxplot(column='Usage_#Camp_CumMoAvg', by='Rating', showfliers=False)
FY14Q3_cohort.boxplot(column='Usage_#DFApptReq_CumMoAvg', by='Rating', showfliers=False)
FY14Q3_cohort.boxplot(column='Usage_#PublicReviews_CumMoAvg', by='Rating', showfliers=False)
FY14Q3_cohort.boxplot(column='Usage_%ApptConfirmRate_Cum', by='Rating', showfliers=False)

FY14Q3_cohort.groupby('Rating')['Usage_%ApptConfirmRate_Cum'].median()
FY14Q3_cohort.groupby('Rating')['Usage_#Camp_CumMoAvg'].median()
FY14Q3_cohort.groupby('Rating')['Usage_#Appts_CumMoAvg'].median()
FY14Q3_cohort.groupby('Rating')['Usage_#DFApptReq_CumMoAvg'].median()
FY14Q3_cohort.groupby('Rating')['Usage_#PublicReviews_CumMoAvg'].median()
FY14Q3_cohort.groupby('Rating')['D3 Public Reviews'].median()


FY14Q3_cohort.boxplot(column='#Customers', by='Rating', showfliers=False)
FY14Q3_cohort.boxplot(column='#Emails', by='Rating', showfliers=False)

FY14Q3_cohort.groupby('Rating')['StackedDiscount_Ind'].mean()
FY14Q3_cohort.groupby('Rating')['D3 Web Alias'].mean()
FY14Q3_cohort.groupby('Rating')['Google+'].mean()
FY14Q3_cohort.groupby('Rating')['D3 Yelp Account'].mean()
FY14Q3_cohort.groupby('Rating')['D3 Facebook Account'].mean()
FY14Q3_cohort.groupby('Rating')['Henry Schein Deal'].mean()

FY14Q3_cohort.groupby('Rating')['#Customers'].median()
FY14Q3_cohort.groupby('Rating')['#Emails'].median()


FY14Q3_cohort.groupby('Rating')['Mgmt_Sys'].value_counts()
 
FY14Q3_cohort['Biz_CustLife'][FY14Q3_cohort['Rating'] == 0].hist(cumulative=True, color='grey')
FY14Q3_cohort['Biz_CustLife'][FY14Q3_cohort['Rating'] == 0].hist()
 
FY14Q3_cohort['Biz_CustLife'].hist(cumulative=True, color='grey')
FY14Q3_cohort['Biz_CustLife'].hist()

# CONSOLIDATED COHORT CHART
FY14Q3_cohort['Biz_CustLife'].hist(cumulative=True, color='grey')
FY14Q3_cohort['Biz_CustLife'][FY14Q3_cohort['Rating'] == 0].hist(cumulative=True, color='red')
FY14Q3_cohort['Biz_CustLife'][FY14Q3_cohort['Rating'] == 1].hist(cumulative=True, color='blue', bins=3)


FY14Q3_cohort.groupby('Rating')
['AccountID', 'Cancellation_Requested_Date'].count() 

FY14Q3_cohort['AccountID'][FY14Q3_cohort['Rating'] == 0][FY14Q3_cohort['Biz_CustLife'] <= 2.0/12 ].count()
FY14Q3_cohort['AccountID'][FY14Q3_cohort['Rating'] == 0][FY14Q3_cohort['Biz_CustLife'] > 2.0/12 ].count()
FY14Q3_cohort['AccountID'][FY14Q3_cohort['Rating'] == 0][FY14Q3_cohort['Biz_CustLife'] < 1 ].count()
FY14Q3_cohort['AccountID'][FY14Q3_cohort['Rating'] == 0][FY14Q3_cohort['Biz_CustLife'] >= 1 ].count()


FY14Q3_cohort.StackedDiscount_Ind.describe()
#how to pull data out of histogram??


''' FEATURE SELECTION '''
all_features = ['D3 Public Reviews', 'Usage_#PublicReviews_CumMoAvg']
# Removes 'Usage_#DFApptReq_CumMoAvg' and 'D3 Web Alias'

sns.pairplot(X)
X.corr()

from scipy.stats import spearmanr, kendalltau, pearsonr
spearmanr(FY14Q3_cohort['D3 Public Reviews'], FY14Q3_cohort['Usage_%ApptConfirmRate_Cum'])


all_features = ['D3 Public Reviews',
'Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 'Usage_#DFApptReq_CumMoAvg', 
'Usage_#PublicReviews_CumMoAvg', 'Usage_%ApptConfirmRate_Cum', 
'#Emails']


all_features = ['StackedDiscount_Ind', 'D3 Public Reviews',
'Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 'Usage_#DFApptReq_CumMoAvg', 
'Usage_#PublicReviews_CumMoAvg', 'Usage_%ApptConfirmRate_Cum', 
'D3 Web Alias', 'Google+', 'D3 Yelp Account', 'D3 Facebook Account',
'#Customers', '#Emails', 'Henry Schein Deal']

['StackedDiscount_Ind', 
'D3 Number of Appointments', 'D3 Number of Campaigns Sent', 'D3 Total Emails Sent', 
'D3 Custom Campaigns Emails Sent', 
'D3 Public Reviews', 'D3 Satisfaction Index', 
'D3 Total Appointments (90 Days)', 'D3 Welcomes Last 60 Days',
'Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 'Usage_#DFApptReq_CumMoAvg', 
'Usage_#PublicReviews_CumMoAvg', 'Usage_%ApptConfirmRate_Cum', 
'D3 Web Alias', 'Google+', 'D3 Yelp Account', 'D3 Facebook Account',
'#Customers', '#Emails', 'Henry Schein Deal']

all_features

X = FY14Q3_cohort[all_features]
y = FY14Q3_cohort.Rating

plt.scatter(X['Usage_#Appts_CumMoAvg'], y, color='red')


from sklearn.feature_selection import chi2
scores, pvalues = chi2(X, y)
pvalues
pd.DataFrame(zip(all_features, pvalues)).sort_index(by=1, ascending=False)


''' TREES '''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import tree
ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
ctree.fit(X_train, y_train)
ctree.classes_
pd.DataFrame(zip(all_features, ctree.feature_importances_)).sort_index(by=1, ascending=False)

from sklearn import metrics
preds = ctree.predict(X_test)
metrics.accuracy_score(y_test, preds)
pd.crosstab(y_test, preds, rownames=['actual'], colnames=['predicted'])
probs = ctree.predict_proba(X_test)[:,1]

metrics.roc_auc_score(y_test, probs)

# Tree graph
tree.export_graphviz(ctree, out_file='tree2.dot')    

from sklearn.externals.six import StringIO
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(ctree, out_file=f)

import os
os.unlink('tree.dot')

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
X_new = clf.fit(X, y).transform(X)
pd.DataFrame(zip(all_features, clf.feature_importances_)).sort_index(by=1, ascending=False)
X_new.shape



''' DECISION TREE CLASSIFIER '''
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

# try max_depth=2
treeclf = DecisionTreeClassifier(max_depth=2, random_state=1)
cross_val_score(treeclf, X, y, cv=10, scoring='roc_auc').mean()

# try max_depth=3
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
cross_val_score(treeclf, X, y, cv=10, scoring='roc_auc').mean()

# use GridSearchCV to automate the search
from sklearn.grid_search import GridSearchCV
treeclf = DecisionTreeClassifier(random_state=1)
max_depth_range = range(1, 21)
param_grid = dict(max_depth=max_depth_range)
grid = GridSearchCV(treeclf, param_grid, cv=10, scoring='roc_auc')
grid.fit(X, y)

# check the results of the grid search
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
grid_mean_scores

# plot the results
import matplotlib.pyplot as plt
plt.plot(max_depth_range, grid_mean_scores)

# what was best?
grid.best_score_
grid.best_params_
grid.best_estimator_


''' MODEL EVALUATION '''
# CROSS VALIDATION
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=8)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
from sklearn.neighbors import KNeighborsClassifier  # import class
knn = KNeighborsClassifier(n_neighbors=5)  

from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc') .mean()
cross_val_score(ctree, X, y, cv=10, scoring='roc_auc').mean()
cross_val_score(nb, X, y, cv=10, scoring='roc_auc').mean()
cross_val_score(knn, X, y, cv=10, scoring='roc_auc').mean()

''' LOGISTIC REGRESSION - 1st PASS '''
FY14Q3_cohort.columns
FY14Q3_cohort['Mgmt_Sys']

feature_cols = ['Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 
'Usage_#DFApptReq_CumMoAvg', 'Usage_#PublicReviews_CumMoAvg', 
'Usage_%ApptConfirmRate_Cum', 'TechSavvy_Google+', '#Customers', 
'#Emails']

'Rating', 'Biz_CustLife', 'Sub_Start_Date'

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.intercept_

zip(all_features, logreg.coef_[0])

y_pred = logreg.predict(X)
from sklearn import metrics
matrix = metrics.confusion_matrix(y, y_pred)

sensitivity = float(matrix[1][1]) / (matrix[1][0] + matrix[1][1])
specificity = float(matrix[0][0]) / (matrix[0][1] + matrix[0][0])
accuracy = (float(matrix[0][0]) + matrix[1][1]) / ((matrix[1][0] + matrix[1][1])+(matrix[0][1] + matrix[0][0]))

from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv=10).mean()
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

# ROC Chart
probs = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

''' MODEL EVALUATION BY COHORT '''

['StackedDiscount_Ind', 'D3 Public Reviews',
'Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 
'Usage_#PublicReviews_CumMoAvg', 'Usage_%ApptConfirmRate_Cum', 
'Google+', 'D3 Yelp Account', 'D3 Facebook Account',
'#Customers', '#Emails', 'Henry Schein Deal']

las_features = ['D3 Public Reviews', 'Usage_#PublicReviews_CumMoAvg']


las_features = ['Usage_#PublicReviews_CumMoAvg', 'D3 Public Reviews']
# Removes 'Usage_#DFApptReq_CumMoAvg' and 'D3 Web Alias'

df_dental.groupby('cohort')['Rating'].count()
X = FY14Q3_cohort[all_features]
y = FY14Q3_cohort.Rating
FY14Q3_cohort.groupby('Rating')['AccountID'].count()

X = FY14Q3_cohort[las_features]
y = FY14Q3_cohort.Rating

cross_val_score(logreg, X, y, cv=10, scoring='roc_auc') .mean()

# FY14 Q2 Cohort
FY14Q2_cohort = df_dental[df_dental['cohort'] == '2014-Q2']
X = FY14Q2_cohort[all_features]
y = FY14Q2_cohort.Rating
FY14Q2_cohort.groupby('Rating')['AccountID'].count()

X = FY14Q2_cohort[las_features]
y = FY14Q2_cohort.Rating

# FY14 Q4 Cohort
FY14Q4_cohort = df_dental[df_dental['cohort'] == '2014-Q4']
X = FY14Q4_cohort[all_features]
y = FY14Q4_cohort.Rating
FY14Q4_cohort.groupby('Rating')['AccountID'].count()

X = FY14Q4_cohort[las_features]
y = FY14Q4_cohort.Rating

# TRAIN TEST SPLIT
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# CROSS VALIDATION
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc') .mean()
cross_val_score(ctree, X, y, cv=10, scoring='roc_auc').mean()
cross_val_score(nb, X, y, cv=10, scoring='roc_auc').mean()
cross_val_score(knn, X, y, cv=10, scoring='roc_auc').mean()


''' REGULARIZATION - LINEAR REGRESSION - LASSO '''
from sklearn.linear_model import LinearRegression, Lasso

las = Lasso(alpha=0.0004, normalize=True)
las.fit(X_train, y_train)
las.coef_
preds = las.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))

zip(all_features, las.coef_)

# use LassoCV to select best alpha (tries 100 alphas by default)
from sklearn.linear_model import LassoCV
lascv = LassoCV(normalize=True)
lascv.fit(X_train, y_train)
lascv.alpha_
lascv.coef_
preds = lascv.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))


''' K-MEANS CLUSTERING OF ACTIVE ACCOUNTS '''
X = FY14Q3_cohort[all_features]
y = FY14Q3_cohort.Rating

# split into train/test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# check that it worked properly
X_train_scaled[:, 0].mean()
X_train_scaled[:, 0].std()
X_train_scaled[:, 1].mean()
X_train_scaled[:, 1].std()

X_train.values
X_train_scaled

# standardize X_test
X_test_scaled = scaler.transform(X_test)

# is this right?
X_test_scaled[:, 0].mean()
X_test_scaled[:, 0].std()
X_test_scaled[:, 1].mean()
X_test_scaled[:, 1].std()

# compare KNN accuracy on original vs scaled data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)

from sklearn.cluster import KMeans
Est = KMeans(n_clusters=3, init='random')
Est.fit(X_train_scaled)
y_kmeans = Est.predict(X_train_scaled)

colors = np.array(['#FF0054','#FBD039','#23C2BC'])
plt.figure()
plt.scatter(X_train['Usage_#Appts_CumMoAvg'], X_train['Usage_%ApptConfirmRate_Cum'], c=colors[y_kmeans], s=50)
plt.xlabel('Usage_#Appts_CumMoAvg')
plt.ylabel('Usage_%ApptConfirmRate_Cum')


 
metrics.silhouette_score(X, Est.labels_, metric='euclidean')   
    
# perform k means with up to 15 clusters
k_rng = range(1,15)
est = [KMeans(n_clusters = k).fit(X) for k in k_rng]

# calculate silhouette score
from sklearn import metrics
silhouette_score = [metrics.silhouette_score(X, e.labels_, metric='euclidean') for e in est[1:]]

# Plot the results
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.plot(k_rng[1:], silhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.plot(7,silhouette_score[5], 'o', markersize=12, markeredgewidth=1.5,
markerfacecolor='None', markeredgecolor='r')

''' LINEAR REGRESSION - CUSTOMER LIFETIME '''
feature_cols = ['Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 
'Usage_#DFApptReq_CumMoAvg', 'Usage_#PublicReviews_CumMoAvg', 
'Usage_%ApptConfirmRate_Cum', 'Google+', '#Customers', 
'#Emails']

all_features = ['StackedDiscount_Ind', 'D3 Public Reviews',
'Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 'Usage_#DFApptReq_CumMoAvg', 
'Usage_#PublicReviews_CumMoAvg', 'Usage_%ApptConfirmRate_Cum', 
'D3 Web Alias', 'Google+', 'D3 Yelp Account', 'D3 Facebook Account',
'#Customers', '#Emails', 'Henry Schein Deal']

['StackedDiscount_Ind', 
'D3 Number of Appointments', 'D3 Number of Campaigns Sent', 'D3 Total Emails Sent', 
'D3 Custom Campaigns Emails Sent', 
'D3 Public Reviews', 'D3 Satisfaction Index', 
'D3 Total Appointments (90 Days)', 'D3 Welcomes Last 60 Days',
'Usage_#Appts_CumMoAvg', 'Usage_#Camp_CumMoAvg', 'Usage_#DFApptReq_CumMoAvg', 
'Usage_#PublicReviews_CumMoAvg', 'Usage_%ApptConfirmRate_Cum', 
'D3 Web Alias', 'Google+', 'D3 Yelp Account', 'D3 Facebook Account',
'#Customers', '#Emails', 'Henry Schein Deal']

all_features

df_dental.columns

X = df_dental[all_features]
y = df_dental.Biz_CustLife

X.isnull().sum()

X.corr()
sns.heatmap(X.corr())
linreg = LinearRegression()
linreg.fit(X, y)
zip(all_features, linreg.coef_)

assorted_pred = linreg.predict(X)

plt.plot(X, assorted_pred, color='red')

''' MORE LINEAR REGRESSION CODE ''' 

# linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_

# make predictions and evaluate
import numpy as np
from sklearn import metrics
preds = lm.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))


''' Appendix - Not Useful For Analysis'''
# Emails Sent / Subscription Life
df_dental['CustEng_#EmailSent_CumMoAvg'] = [float(a)/b for a,b in zip(df_dental['D3 Total Emails Sent'], df_dental['MonthsUnderContract'])]
df_dental['CustEng_#EmailSent_CumMoAvg'].describe()
df_dental.groupby('Rating')['CustEng_#EmailSent_CumMoAvg'].describe()

# Custom Campaigns Sent / Subscription Life
df_dental['CustEng_#CustomCamp_CumMoAvg'] = [float(a)/b for a,b in zip(df_dental['D3 Custom Campaigns Emails Sent'], df_dental['MonthsUnderContract'])]
df_dental['CustEng_#CustomCamp_CumMoAvg'].describe()
df_dental.groupby('Rating')['CustEng_#CustomCamp_CumMoAvg'].describe()
df_dental.groupby('MonthsUnderContract')['CustEng_#CustomCamp_CumMoAvg'].describe()

# Emails Sent / Campaign - [Need to group by account]
df_dental['CustEng_#EmailSent_Campaign'] = [float(a)/b for a,b in zip(df_dental['D3 Total Emails Sent'], df_dental['D3 Number of Campaigns Sent'])]
df_dental['CustEng_#EmailSent_Campaign'].describe()
df_dental.groupby('Rating')['CustEng_#EmailSent_Campaign'].describe()


# % of dental accounts w/ email enabled - can be disabled shortly prior to cxl date; no time series
D3_Email_Enabled = [x for x in df_dental['D3 Email Enabled'] if x == 1]
float(sum(D3_Email_Enabled)) / df_dental.shape[0]

# Custom metric - % of Total Appointments Confirmed (Last 90 Days)
df_dental['ValOM_%ApptConfirmRate_90Day']  = [float(a)/b for a,b in zip(df_dental['D3 Confirmed Appointments (90 Days)'],df_dental['D3 Total Appointments (90 Days)'])]
df_dental['ValOM_%ApptConfirmRate_90Day'].describe()

# % of DF Appointments Confirmed (Last 90 Days)
df_dental['ValOM_%DFApptConfirmRate_90Day']  = [float(a)/b for a,b in zip(df_dental['D3 Confirmed Appointments (90 Days)'],df_dental['D3 Total Appointments (90 Days)'])]
df_dental['ValOM_%DFApptConfirmRate_90Day'].describe()

# Custom metric - % of Total Visits / Total Emails Sent
df_dental['ValROI_%TotVisit-to-EmailSent_Cum']  = [float(a)/b * 100 for a,b in zip(df_dental['D3 Total Visits'],df_dental['D3 Total Emails Sent'])]
df_dental['ValROI_%TotVisit-to-EmailSent_Cum'].describe()

# last login
def CustEng_LastLogin_Days_calc(row):
    if row.Rating == 'Under Contract':
        return present - row['D3 Last Login Date to Portal']
    return row.Cancellation_Requested_Date - row['D3 Last Login Date to Portal']

df_dental['CustEng_LastLogin_Days'] = df_dental.apply(CustEng_LastLogin_Days_calc, axis=1)
df_dental['CustEng_LastLogin_Days'].isnull().sum()


''' UNUSED CODE '''

FY14Q3_cohort['Rating'][FY14Q3_cohort['Biz_CustLife'] <= 0.5].shape[0] * 1.0 / FY14Q3_cohort.shape[0]
FY14Q3_cohort['Rating'][FY14Q3_cohort['Biz_CustLife'] <= 1.0].shape[0] * 1.0 / FY14Q3_cohort.shape[0]
FY14Q3_cohort['Rating'][FY14Q3_cohort['Biz_CustLife'] <= 1.5].shape[0] * 1.0 / FY14Q3_cohort.shape[0]

FY14Q3_cohort[(FY14Q3_cohort.Rating=='Cancelled')].groupby('Cancellation_Requested_Date').Biz_CustLife.count().hist()


# Cancelled base within last 3 yrs
df_dental[df_dental.Biz_CustLife].shape[0]

df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 3][df_dental.Rating == 'Cancelled'].describe()
df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 3][df_dental.Rating == 'Cancelled'].shape[0]
df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 3][df_dental.Rating == 'Cancelled'].shape[0] / (df_dental.Biz_CustLife.shape[0] * 1.0)

# Cancelled base within last 2 yrs
df_dental[df_dental.Biz_CustLife <= 2].shape[0]

df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 2][df_dental.Rating == 'Cancelled'].describe()
df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 2][df_dental.Rating == 'Cancelled'].shape[0]
df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 2][df_dental.Rating == 'Cancelled'].shape[0] / (df_dental.Biz_CustLife.shape[0] * 1.0)

# Cancelled base within last 1 yr
df_dental[df_dental.Biz_CustLife <= 1].shape[0]

df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 1][df_dental.Rating == 'Cancelled'].describe()
df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 1][df_dental.Rating == 'Cancelled'].shape[0]
df_dental['Henry Schein Deal'][df_dental.Biz_CustLife <= 1][df_dental.Rating == 'Cancelled'].shape[0] / (df_dental.Biz_CustLife.shape[0] * 1.0)
    

''' CHARTS! '''
# Active Accounts
plt.figure()  
plt.axis([0, 12, 0, 700])  

binBoundaries = np.linspace(0,20,80)

df_dental['Biz_CustLife'][df_dental.Rating == 'Under Contract'].hist(bins=binBoundaries, label='Active')

plt.title('Customer Lifetime')  
plt.xlabel('Customer Lifetime (Yrs)')  
plt.ylabel('# of Accounts')  
plt.legend()

# Cancelled Accounts
plt.figure()  
plt.axis([0, 12, 0, 2000])  

binBoundaries = np.linspace(0,20,80)

df_dental['Biz_CustLife'][df_dental.Rating == 'Cancelled'].hist(bins=binBoundaries, label='Cancelled') 

plt.title('Customer Lifetime')  
plt.xlabel('Customer Lifetime (Yrs)')  
plt.ylabel('# of Accounts')  
plt.legend()