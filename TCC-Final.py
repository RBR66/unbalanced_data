#!/usr/bin/env python
# coding: utf-8

# ##  MBA Ciência de Dados
# ### Tratamento de dados desbalanceados em classificação binária com algoritmos em Python e aplicação em Marketing
# #### Ramon Barbosa Rosa
# ##### Dezembro 2020

# In[1]:


#!pip install -U scikit-learn
#!pip install openpyxl


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import timeit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from scipy import stats #Testes de hipóteses
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline


# In[3]:


path = r'D:\Documentos D\ESTUDOS\Arquivos MBA Ciência de Dados - USP (G Drive)\12-TCC\Datasets\bank-additional (UCI)'
path_for_save = r'D:\Documentos D\ESTUDOS\Arquivos MBA Ciência de Dados - USP (G Drive)\12-TCC'
file = r'\bank-additional-full.xlsx'
data = pd.read_excel(f'{path}{file}', engine='openpyxl')
data.columns = map(str.upper, data.columns)
data.head()


# # 1 - Descriptive analysis and data preprocessing

# ## 1.1 - Data general info

# In[4]:


data.info()


# ## 1.2 - Convert target variable (y) to binary

# In[5]:


#Convert 'y' (label) to numeric
data.rename(columns={'Y': 'TARGET'}, inplace=True) 
rule = {'yes':1, 'no':0}
data['TARGET'].replace(rule, inplace=True)


# ## 1.3 - Descriptive statistcs

# In[6]:


descriptive_before = round(data.describe().T, 3)
descriptive_before.to_excel(f'{path_for_save}/descriptive before.xlsx', sheet_name='descriptive before')
descriptive_before


# ## 1.4 - Analyze numeric variables

# In[7]:


info = pd.DataFrame(data.dtypes)
numeric_cols = [i for i in info.index if info.loc[i][0] != 'object']
numeric_cols = numeric_cols[:-1] #Remove TARGET from list of variables
print('Numeric variables: ', 'len=', len(numeric_cols), numeric_cols)
print()


# ### 1.4.1 - Histogram for numeric variables

# In[8]:


#Histogram for numeric variables
nrows, ncols, counter = 4, 3, 0

axes = data[numeric_cols].hist(layout=(nrows, ncols), figsize=(30,20), bins=20, color=(33/256,145/256,140/256))
for i in range(nrows): 
    for j in range(ncols):
        if counter < len(numeric_cols):
            col = numeric_cols[counter]
            axes[i, j].set_title(col, fontsize=20)
            counter += 1
        else: break
img_name = r'\Histogram for numeric variables_before'
plt.savefig(f'{path_for_save}{img_name}.jpg', dpi=800)
plt.show()


# ### 1.4.2 - Box-plot against target for numeric variables

# In[9]:


#Box-plot para cada predictor com o TARGET
nrows = 3
ncols = 4
counter = 0

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,23))
for i in range(nrows):
    for j in range(ncols):
        if counter < len(numeric_cols):
            col = numeric_cols[counter]  
            TARGET0 = data[data['TARGET']==0][col]
            TARGET1 = data[data['TARGET']==1][col]
            ax[i, j].boxplot(x=[TARGET0, TARGET1], labels=('TARGET=0','TARGET=1'), showmeans=True)
            ax[i, j].set_title(col.upper(), fontsize=20)
            counter += 1
        else: break
img_name = r'\Box-plot_before'
plt.savefig(f'{path_for_save}{img_name}.jpeg', dpi=800)
plt.subplots_adjust(wspace=0.5, hspace=0.3)


# ### 1.4.3 - Student's t-test for inference about means of each predictor on classes of the TARGET

# In[10]:


#Separando a base em dois grupos: que fez e quem não fez o investimento
data_0 = data[numeric_cols][data['TARGET']== 0] #Grupo que não fez o investimento
data_1 = data[numeric_cols][data['TARGET']== 1] #Grupo que fez o investimento
print('Number of samples for each class')
print('Shape data_0', data_0.shape)
print('Shape data_1', data_1.shape)


# In[11]:


# Teste t de Student para igualdade de médias (supondo amostras independentes e variâncias iguais)
# H0: as médias são iguais, portanto não há diferença entre os grupos.
# H1: as médias são diferentes, indicando que há diferença entre os grupos, na variável em questão.

results = pd.DataFrame(columns = ['Variável','Estatística do teste','p-value', 'Resultado'], 
                         index = np.arange(data.shape[1]))

threshold = 0.05

for i, (varn, vary) in enumerate(zip(data_0, data_1)):
    t_test = stats.ttest_ind(data_0[varn], data_1[vary], equal_var=False) 
    if t_test[1] <= threshold:
        result = "Rejeita H0"
    else:
        result = "Aceita H0"
    
    results.iloc[i,0] = varn #Nome da variável
    results.iloc[i,1] = round(t_test[0], 3) #Estatística do teste
    results.iloc[i,2] = round(t_test[1], 3) #p-value
    results.iloc[i,3] = result

results.to_excel(f'{path_for_save}/student test before.xlsx', sheet_name='student test before')
results[:10]


# ### 1.4.4 - Correlation analysis of numeric variables

# In[12]:


plt.figure(figsize=(10,10))
annot_kws={'fontsize':14}
low_matrix = np.triu(data[numeric_cols].corr())
ax = sns.heatmap(data[numeric_cols].corr(), annot=True, fmt='.3f', cmap='viridis', square=True, linewidths=.5, mask=low_matrix, annot_kws=annot_kws)
plt.title('Correlation Matrix of raw numeric variables', fontsize=15);
img_name = r'\Correlations_before'
plt.savefig(f'{path_for_save}{img_name}.jpeg', dpi=1600)


# ## 1.5 - Categoric variables

# ### 1.5.1 - Distribution of classes on categoric variables 

# In[13]:


info = pd.DataFrame(data.dtypes)
categoric_cols = [i for i in info.index if info.loc[i][0] == 'object']
print('Categoric variables: ', 'len=', len(categoric_cols), categoric_cols)
print()

for col in categoric_cols:
    ind = data[col].value_counts().index
    val = data[col].value_counts().values
    var_object = pd.DataFrame(data={'Values':ind, 'Qtde':val})
    var_object['Pct'] = [v/var_object['Qtde'].sum() for v in var_object['Qtde']]


# ### 1.5.2 - Analyze categoric variables and its relation with TARGET variable

# In[14]:


chi_sqr_results = pd.DataFrame()
categoric_cols


# In[15]:


for cat_var in categoric_cols:
    #Chi Square test for testing for TARGET diferences among categories of JOB variable
    #HO: there is no impact of this variable on TARGET (no difference of distributions)
    contigency= pd.crosstab(data.TARGET, data[cat_var], normalize='columns') 
    c, p, dof, expected = chi2_contingency(contigency)
    alpha = 0.05
    result = 'Rejeita HO' if (p <= alpha) else 'Aceita H0'
    table= pd.Series({'Chi Square statistic':round(c, 3), 'p-value':round(p, 3), 'Degrees of Freedom':dof, 'Result':result})
    chi_sqr_results[cat_var] = table

    #Plotting for visualize diferences on the TARGET proportion among categories
    #Preparing data
    df = pd.crosstab(data[cat_var], data.TARGET, margins=True)
    df['% 0'] = df[0]/df['All']
    df['% 1'] = df[1]/df['All']  

    #Plotting
    ax  = df.iloc[:, 3:5].plot.bar(rot=45, figsize=(15,8), title=cat_var, ylim=(0,1.2), 
                               color={'% 0': (51/256, 98/256, 141/256), '% 1': (200/256, 224/256, 32/256)}, xlabel = '',
                               ylabel='Percentage in each TARGET class', stacked=False);
    for p in ax.patches:
        ax.annotate(str(format(p.get_height(), '.0%')), (p.get_x() * 1.005, p.get_height() * 1.005))

    #Save for file
    img_name = f'\{cat_var}_before'
    plt.savefig(f'{path_for_save}{img_name}.jpeg', dpi=1600)


# In[16]:


#Print results of Chi Square tests within all categoric variables
chi_sqr_results.to_excel(f'{path_for_save}/chi_sqr_results test before.xlsx', sheet_name='chi sqr results')
chi_sqr_results


# # 2 - Data Wrangling

# ## 2.1 - Feature selection

# From the EDA stage we select predictors of interest.

# In[17]:


data.columns


# In[18]:


selected_predictors = ['DURATION', 'EURIBOR3M','CONS.CONF.IDX' 
                       ,'JOB', 'DEFAULT', 'CONTACT', 'MONTH', 'POUTCOME', 'TARGET']
# Obs.TARGET is not a predictor


# ## 2.1 - Split train and test sets

# In[19]:


#Preparing data - 20% para teste
data_new = data[selected_predictors]
#data_new = data_new.sample(frac=0.1, replace=False, random_state=1) 
X = data_new[data_new.columns[:-1]].copy()
y = data_new[data_new.columns[-1:]].copy()
print('X shape', X.shape)
print('y shape', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

zeros_train = y_train[y_train['TARGET']==0].count() / y_train['TARGET'].count()
ones_train = y_train[y_train['TARGET']==1].count() / y_train['TARGET'].count()
zeros_test = y_test[y_test['TARGET']==0].count() / y_test['TARGET'].count()
ones_test = y_test[y_test['TARGET']==1].count() / y_test['TARGET'].count()

print('X_train Shape = ', X_train.shape, ' % 0 = ', zeros_train, ' % 1 = ', ones_train)
print('X_test Shape = ', X_test.shape, ' % 0 = ', zeros_test, ' % 1 = ', ones_test)
print('y_train shape', y_train.shape)
print('y_test shape' ,y_test.shape)


# ## 2.2 - Transformation, convertions and scaling

# In[20]:


cols_to_transform = ['DURATION', 'EURIBOR3M', 'CONS.CONF.IDX']
cols_to_scale = ['DURATION', 'EURIBOR3M', 'CONS.CONF.IDX']


# ### 2.2.1 - Transformation of continuous variables 

# In[21]:


# Function for transforming data

def func_transform(X_train, X_test, list_of_cols_to_transform):
    for data_set in [X_train, X_test]:
        for col in data_set.columns:
            if col in list_of_cols_to_transform:
                col_to_transform = np.array(data_set[col])
                #col_transformed = np.log(col_to_transform + 0.001)
                col_transformed = np.sqrt(col_to_transform + np.abs(np.min(col_to_transform))) 
                #col_transformed = col_to_transform**(1/3)
                #col_transformed = 1/col_to_transform 
                data_set.loc[:, col] = col_transformed
    return X_train, X_test


# ### 2.2.2 - Convert text variables to numeric

# In[22]:


# Function to convert text variables to numeric

def func_convert_to_numeric(X_train, X_test):
    for data_set in [X_train, X_test]: 
        
        #Convert 'job' to numeric
        jobs_to_one = ['student', 'retired']
        rule1 = {j: 1 for j in data_set['JOB'].unique() if j in jobs_to_one}
        rule2 = {j: 0 for j in data_set['JOB'].unique() if j not in jobs_to_one}
        rule = {**rule1, **rule2} #Concat two dicts
        data_set['JOB'].replace(to_replace=rule, inplace=True)

        #Convert 'default' to numeric
        rule = {'no':1, 'yes':0, 'unknown':0}
        data_set['DEFAULT'].replace(to_replace=rule, inplace=True)
        
        #Convert 'contact' to numeric
        rule = {'cellular':1, 'telephone':0}
        data_set['CONTACT'].replace(to_replace=rule, inplace=True)  
        
        #Convert 'month' to numeric, join with orginal data and remove object column
        months_to_one = ['dec','mar','oct','sep']
        rule1 = {m : 1 for m in data_set['MONTH'].unique() if m in months_to_one}
        rule2 = {m : 0 for m in data_set['MONTH'].unique() if m not in months_to_one}
        rule = {**rule1, **rule2} #Concat two dicts
        data_set['MONTH'].replace(to_replace=rule, inplace=True)

        #Convert 'poutcome' to numeric
        rule = {'nonexistent':0, 'failure':0, 'success':1}
        data_set['POUTCOME'].replace(to_replace=rule, inplace=True)  
    
    return X_train, X_test


# ### 2.2.3 - Feature scaling of some predictors

# In[23]:


# Function for standardize some variables

def func_std_scaler(X_train, X_test, cols_to_scale):
    for c in cols_to_scale:
        #Compute standardization
        var_mean = X_train[c].mean()
        var_std = X_train[c].std()
        var_norm_X_train = (X_train[c] - var_mean)/var_std
        var_norm_X_test = (X_test[c] - var_mean)/var_std
        #Drop raw variables
        X_train.drop(c, axis=1)
        X_test.drop(c, axis=1)
        #Assign standardized varibles to data frames
        X_train[c] = var_norm_X_train
        X_test[c] = var_norm_X_test       
        #X_train.loc[:, c] = (X_train[c] - m)/s
        #X_test.loc[:, c] = (X_test[c] - m)/s
    return X_train, X_test

test = func_std_scaler(X_train, X_test, cols_to_scale)


# ### 2.2.4 - Execute transformation, conversion and scaling

# In[24]:


#Function for call alt transformations - all sets                         
def func_pipeline(X_train, X_test):
    functions = [
                 func_transform(X_train, X_test, cols_to_transform),
                 func_convert_to_numeric(X_train, X_test),      
                 func_std_scaler(X_train, X_test, cols_to_scale)
                ]       
    for func in functions:
        X_train_prepared, X_test_prepared = X_train, X_test
    return X_train_prepared, X_test_prepared 
    
# Call transformations and rename data sets to better identify them before and after transformations
X_train_prepared, X_test_prepared = func_pipeline(X_train, X_test)


# In[25]:


print()
print(X_train_prepared.info())
print()
print(X_test_prepared.info())


# ## 2.6 - Descriptive statistics after all data wrangling (training set)

# In[26]:


descriptive_after = round(X_train_prepared.describe().T, 3)
descriptive_after.to_excel(f'{path_for_save}/descriptive after.xlsx', sheet_name='descriptive after')
descriptive_after 


# ## 2.7 - Correlation analysis after all data wrangling (training set)

# In[27]:


plt.figure(figsize=(10,10))
annot_kws={'fontsize':14}
low_matrix = np.triu(X_train_prepared.corr())
ax = sns.heatmap(X_train_prepared.corr(), annot=True, fmt='.3f', cmap='viridis', square=True, linewidths=.5, mask=low_matrix, annot_kws=annot_kws)
plt.title('Correlation Matrix after all data wrangling (training set)', fontsize=15);
img_name = r'\Correlations_after'
plt.savefig(f'{path_for_save}{img_name}.jpeg', dpi=1600)


# ## 2.8 - Graphical analysis of remained predictors

# ### 2.8.1 - Histogram of numerical variables

# In[43]:


nrows, ncols, counter = 1, 3, 0
cols = X_train_prepared.columns[:3]
axes = X_train_prepared[cols].hist(layout=(nrows, ncols), figsize=(30,8), bins=20, color=(33/256,145/256,140/256))
      
for i, col in enumerate(cols): 
    if counter < len(cols):
        col = X_train_prepared.columns[counter]
        axes[i].set_title(col, fontsize=20)
        counter += 1
    else: break
            
img_name = r'\Histogram for numeric variables_after'
plt.savefig(f'{path_for_save}{img_name}.jpg', dpi=800)
plt.show()


# ### 2.8.1 - Graphs of categoric variables

# In[ ]:



labels = 'Zeros', 'Ones'
sizes = [15, 30, 45, 10]
explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(layout=(2,3), figsize=(30,20))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ## 2.9 - Box-Plot against the TARGET for the final predictors (training set)

# In[65]:


#Box-plot para cada predictor com o TARGET
nrows, ncols, counter = 3, 3, 0

final_predictors = X_train_prepared.columns
final_data = X_train_prepared.join(y_train, how='outer')

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,23))
for i in range(nrows):
    for j in range(ncols):
        if counter < len(final_predictors):
            col = final_predictors[counter]  
            target0 = final_data[final_data['TARGET']==0][col]
            target1 = final_data[final_data['TARGET']==1][col]
            ax[i, j].boxplot(x=[target0, target1], labels=('Target=0','Target=1'), showmeans=True)
            ax[i, j].set_title(col.upper(), size=20)
            counter += 1
        else: break
            
img_name = r'\Box-plot_after'
plt.savefig(f'{path_for_save}{img_name}.jpeg', dpi=800)            


# # 3 - Modelling

# ### 3.1 - Logistic Regression (Baseline model)

# In[66]:


#Baseline classifier 
from sklearn.linear_model import LogisticRegression

#Model selection methods
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

#Performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
from imblearn.metrics import geometric_mean_score

from scipy import stats


# In[67]:


# Raveling TARGET sets
#y_train =  y_train.ravel()
#y_test = y_test.ravel()
print(y_train.shape)
print(y_test.shape)
y_train =  y_train.to_numpy(copy=True).ravel()
y_test = y_test.to_numpy(copy=True).ravel()

#Define the baseline model (no hyperparameters)
LR = LogisticRegression() 
LR.fit(X_train_prepared, y_train)

#Since whe are not interested in compare diferent raw models, whe do not load cross_val_score
#LR_scores = cross_val_score(LR, X_test_prepared, y_test, cv=rskf, scoring='roc_auc', n_jobs=-1)

#Set some parameters for GridSearch with Logistic Regression
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
param_grid = [{'penalty': ['none'], 'solver':['newton-cg', 'lbfgs', 'sag','saga']},
              {'penalty': ['l1'], 'C': [0.1, 1, 10, 100, 1000], 'solver':['liblinear', 'saga']},
              {'penalty': ['l2'], 'C': [0.1, 1, 10, 100, 1000], 'solver':['newton-cg', 'lbfgs', 'sag', 'saga']}]

#Define and fit GridSearch function with ross-validation
grid_search = GridSearchCV(LR, param_grid, cv=rskf, scoring='roc_auc', return_train_score=True)
grid_search.fit(X_train_prepared, y_train)
LR_best_model = grid_search.best_estimator_
print('LR Best parameters: ', grid_search.best_params_)

#Evaluate the best model in the test set
y_pred = LR_best_model.predict(X_test_prepared)
probs = LR_best_model.predict_proba(X_test_prepared)
print(len(y_pred))
print(len(probs))
print(X_test_prepared.shape)
     
#Main Metrics
def some_metrics(cm):
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    ACC = (TP + TN)/(TP + TN + FP + FN)
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    CSI = TP/(TP + FP + FN)
    GS = (TP*TN - FP*FN)/((FN + FP)*(TP + FP + FN + TN) + (TP*TN - FP*FN))
    SSI = TP/(TP + 2*FP + 2*FN)
    FAITH = (TP + 0.5*TN)/(TP + FP + FN + FN)
    PDIF = (4*FP*FN)/(TP + FP + FN + TN)**2
    return ACC, TPR, TNR, CSI, GS, SSI, FAITH, PDIF

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('Threshold')
    plt.legend(labels=['Precision', 'Recall'], loc='best')
    plt.title('Precision-Recall Curve - LR')
    
cm = confusion_matrix(y_test, y_pred)
sm = some_metrics(cm)

#Precision-Recall Curve (Géron 2019, pg 143)
#precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.show()

#Save image for file
img_name = r'\baseline log reg precision recall'
plt.savefig(f'{path_for_save}{img_name}.jpeg', dpi=1600) 

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1.2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')

img_name = r'\baseline roc curve_before'
plt.savefig(f'{path_for_save}{img_name}.jpeg', dpi=1600)     
    
plot_roc_curve(fpr, tpr)
plt.show()


# In[70]:


#Summarizing the baseline model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.tools as stt

model = sm.Logit(y_train, X_train_prepared) #Defining the model as logistic Regression
results = model.fit() #Fiting the model and creating an instance (object) with the results. Within this object resides
                      #all elements of the summary that can be retrieved separetelly.
summary = results.summary() #Calling the complete table with results
#summary.to_excel(f'{path_for_save}/baseline logistic reg summary.xlsx', sheet_name='summary')
#csvfile = '\baseline logistic reg summary.csv'
#summary.as_csv(f'{path_for_save}{csvfile}')
summary


# ## 3.2 - Imbalanced models

# ### 3.2.1 - Import models and define function for performance metrics

# In[71]:


#Over-sampling algorithms
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN

#Under-sampling algorithms
from imblearn.under_sampling import TomekLinks #https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.TomekLinks.html#imblearn.under_sampling.TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold

#Combination of over- and under-sampling
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

#Ensemble classifiers
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier

#Miscelaneous
from collections import Counter


# In[72]:


#Metrics
def func_metrics(y_test, y_test_pred, probs, X, y):

    # Summarize the new classes distribution
    lenght = len(X)
    classes_size = sorted(Counter(y).items())
    
    #Metrics native from Scikit Learn
    PRC = precision_score(y_test, y_test_pred) #Precision
    REC = recall_score(y_test, y_test_pred) #Recall, Specificit or True Positive Rate
    F1S = f1_score(y_test, y_test_pred) #F1 Score
    AUC = roc_auc_score(y_test, probs) #ROC auc score   
    
    #Metrics from Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    ACC = (TP + TN)/(TP + TN + FP + FN) #Accuracy
    TPR = TP/(TP + FN) #Sensitivity - Recall - True Positive Rate
    TNR = TN/(TN + FP) #True Negative Rate
    CSI = TP/(TP + FP + FN) #Jaccard Index
    GSS = (TP*TN - FP*FN)/((FN + FP)*(TP + FP + FN + TN) + (TP*TN - FP*FN)) #Gilbert Skill Score 
    SSI = TP/(TP + 2*FP + 2*FN) #Sokal & Sneath Index
    FAITH = (TP + 0.5*TN)/(TP + FP + FN + FN) #Faith Index
    PDIF = (4*FP*FN)/(TP + FP + FN + TN)**2 #Pattern Difference
    
    return lenght, classes_size, ACC, PRC, REC, TNR, TPR, F1S, AUC, CSI, GSS, SSI, FAITH, PDIF 


# ### 3.2.2 - Defining models and calculate its performance metrics on test set

# In[73]:


# Defining models
SMOT = SMOTE()
SVMSMOT = SVMSMOTE()
ADASY = ADASYN()
TOMEK = TomekLinks()
NMISS = NearMiss()
ENN = EditedNearestNeighbours()
RENN = RepeatedEditedNearestNeighbours()
AKNN = AllKNN()
CNN = CondensedNearestNeighbour(sampling_strategy='not minority', random_state=42, n_neighbors=3)
OSS = OneSidedSelection()
NCR = NeighbourhoodCleaningRule()
IHT = InstanceHardnessThreshold()
SMOTET = SMOTETomek()
SMOTEEN = SMOTEENN()
EEC = EasyEnsembleClassifier(n_jobs=-1)
BRFC = BalancedRandomForestClassifier(n_jobs=-1)
RUSBC = RUSBoostClassifier()

models_name = ['SMOTE','SVMSMOTE','ADASYN','TOMEK','NM','ENN','RENN','AllKNN', 'OSS','NCR','IHT','SMOTET','SMOTEENN','EEC','BRFC','RUSBC']
models_instance = [SMOT, SVMSMOT, ADASY, TOMEK, NMISS, ENN, RENN, AKNN, OSS, NCR, IHT, SMOTET, SMOTEEN, EEC, BRFC, RUSBC]

list_models_metrics = ['Model','Set Size','classes_size','ACC','PRC','REC', 'TNR', 'TPR', 'F1S','AUC','CSI','GSS','SSI','FAITH','PDIF']
metrics_panel = pd.DataFrame(columns=list_models_metrics)
time_panel = pd.DataFrame(columns= ['Log Reg', 'Log Reg Tuned'] + models_name)

#Evaluate the Logistic Regression best model in the test set
lr_models = [LR, LR_best_model]
lr_models_name = ['Log Reg', 'Log Reg Tuned']
LR = LogisticRegression() 

start_row = metrics_panel.shape[0]
for row, (model_name, model_instance) in enumerate(zip(lr_models_name, lr_models)):
    start_time = timeit.default_timer()
    model_instance.fit(X_train_prepared, y_train)
    y_test_pred = model_instance.predict(X_test_prepared)
    probs = model_instance.predict_proba(X_test_prepared)[:,1]
    metrics = func_metrics(y_test, y_test_pred, probs, X_test_prepared, y_test)
    for column, metric in enumerate(metrics):
        metrics_panel.loc[row+start_row, 'Model'] = model_name
        metrics_panel.iloc[row+start_row, column+1] = metric       
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    time_panel.loc[0, model_name] = total_time

# Inbalanced methods (over sampling, under sampling and combination of over- and under-sampling)
start_row = metrics_panel.shape[0]
model_lr = lr_models[1] #Using the tuned Logistic Regression Model with ressampled data
for row, (model_name, model_instance) in enumerate(zip(models_name[:13], models_instance[:13])):
    start_time = timeit.default_timer()
    X_resampled, y_resampled = model_instance.fit_resample(X_train_prepared, y_train)  
    model_resampled = model_lr.fit(X_resampled, y_resampled)
    y_test_pred = model_resampled.predict(X_test_prepared) 
    probs = model_resampled.predict_proba(X_test_prepared)[:,1] #Get only positive values 
    cm = confusion_matrix(y_test, y_test_pred)
    metrics = func_metrics(y_test, y_test_pred, probs, X_resampled, y_resampled)
    for column, metric in enumerate(metrics):
        metrics_panel.loc[row+start_row, 'Model'] = model_name
        metrics_panel.iloc[row+start_row, column+1] = metric        
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    time_panel.loc[0, model_name] = total_time
        
#Ensemble Classifiers
start_row = metrics_panel.shape[0]
for row, (model_name, model_instance) in enumerate(zip(models_name[13:], models_instance[13:])):
    model_ressampled = model_instance.fit(X_train_prepared, y_train)
    y_pred = model_ressampled.predict(X_test_prepared) 
    probs = model_resampled.predict_proba(X_test_prepared)[:,1]
    #y_test_pred_geron = cross_val_predict(estimator=model_resampled, X=X_test_prepared, y=y_test, cv=3, n_jobs=-1)
    #y_test_scores = cross_val_predict(estimator=model_ressampled, X=X_test_prepared, y=y_test, cv=5, method="decision_function", n_jobs=-1)  
    metrics = func_metrics(y_test, y_pred, probs, X, y)
    for column, metric in enumerate(metrics):
        metrics_panel.loc[row+start_row, 'Model'] = model_name
        metrics_panel.iloc[row+start_row, column+1] = metric
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    time_panel.loc[0, model_name] = total_time

#Save for file    
metrics_panel.to_excel(f'{path_for_save}/Metrics panel.xlsx', sheet_name='Metrics panel')
time_panel.to_excel(f'{path_for_save}/Time panel.xlsx', sheet_name='Time panel')


# ### 3.2.3 - Panel of performance metrics

# In[74]:


metrics_panel


# ### 3.2.4 - Panel of time for processing each model

# In[75]:


time_panel


# ### 3.2.5 - Feature importance

# In[76]:


#Generate feature importance from a choose model
model_for_feature_importance = BRFC
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X_train_prepared.columns, model_for_feature_importance.feature_importances_):
    feats[feature] = importance #add the name/value pair 

#Save feature importance in a data frame and creat a graph
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Feature importance'})
ax = importances.sort_values(by='Feature importance', ascending=False).plot(kind='bar', rot=45, color=(33/256,145/256,140/256))
for p in ax.patches:
    ax.annotate(str(format(p.get_height(), '.0%')), (p.get_x() * 1.005, p.get_height() * 1.010))

#Save for file    
img_name = r'\Feature importance'
plt.savefig(f'{path_for_save}{img_name}.jpg', dpi=1600)

importances


# ## References
# * MCKINNEY, Wes - Python for Data Analysis - O'Reilly - 2018
# * GÉRON, Aurelien - Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow - O'Reilly - 2019
# * LEMAITRE, Guilhaume. NOGUEIRA, Fernando. ARIDAS, Christos K., Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning, Journal of Machine Learning Research - 2017

# In[ ]:




