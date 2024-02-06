# %% [markdown]
# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# %% [markdown]
# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# %matplotlib inline
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, chi2

player_df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Applied AI assgn\ML 1\fifa19.csv")
# player_df = player_df.drop('Nationality', axis=1)

numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']

player_df = player_df[numcols+catcols]

traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()

traindf = pd.DataFrame(traindf,columns=features)

y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']

X.head()

len(X.columns)

# ### Set some fixed set of features
feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30

# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
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
# Your code ends here
    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')

# ### List the selected features from Pearson Correlation

cor_feature

# ## Filter Feature Selection - Chi-Sqaure

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# ### Chi-Squared Selector function

# %%
def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    chi2_selector = SelectKBest(chi2, k=num_feats)
    
    chi2_selector.fit(X, y)
    chi_support = chi2_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    # Your code ends here
    return chi_support, chi_feature

# %%
chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')

# %% [markdown]
# ### List the selected features from Chi-Square 

# %%
chi_feature

# %% [markdown]
# ## Wrapper Feature Selection - Recursive Feature Elimination

# %%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# ### RFE Selector function

# %%
def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    model = LogisticRegression(solver='liblinear')
    rfe_selector = RFE(model, n_features_to_select=num_feats)
    rfe_selector = rfe_selector.fit(X, y)
    
    # Get the selected features and their support
    rfe_support = rfe_selector.support_
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature

# %%
rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')

# %% [markdown]
# ### List the selected features from RFE

# %%
rfe_feature

# %% [markdown]
# ## Embedded Selection - Lasso: SelectFromModel

# %%
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# %%
def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    model = LogisticRegression(penalty='l1', solver='liblinear')
    embedded_lr_selector = SelectFromModel(model, max_features=num_feats)
    embedded_lr_selector.fit(X, y)
    
    # Get the selected features and their support
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature

# %%
embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')

# %%
embedded_lr_feature

# %% [markdown]
# ## Tree based(Random Forest): SelectFromModel

# %%
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# %%
def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Using Random Forest as the estimator for feature selection
    model = RandomForestClassifier(n_estimators=100)
    embedded_rf_selector = SelectFromModel(model, max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    
    # Get the selected features and their support
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature

# %%
embedder_rf_support, embedder_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedder_rf_feature)), 'selected features')

# %%
embedder_rf_feature

# %% [markdown]
# ## Tree based(Light GBM): SelectFromModel

# %%
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

# %%
def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
     # Using Light GBM as the estimator for feature selection
    model = LGBMClassifier(n_estimators=100)
    embedded_lgbm_selector = SelectFromModel(model, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    
    # Get the selected features and their support
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature

# %%
embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')

# %%
embedded_lgbm_feature

# %% [markdown]
# ## Putting all of it together: AutoFeatureSelector Tool

# %%
pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedder_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df.iloc[:,1:].astype(int), axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)

# %% [markdown]
# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# %%
def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    X = player_df.iloc[:, :-1]
    y = player_df.iloc[:, -1]
    
    num_feats = X.shape[1]
    
    return X, y, num_feats
    # Your code ends here
    return X, y, num_feats

# %%
def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    best_features=[]
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    # X, y, num_feats = preprocess_dataset(dataset_path)

    # # Run each feature selection method
    # cor_support, _ = cor_selector(X, y, num_feats) if 'pearson' in methods else (None, None)
    # chi_support, _ = chi_squared_selector(X, y, num_feats) if 'chi-square' in methods else (None, None)
    # rfe_support, _ = rfe_selector(X, y, num_feats) if 'rfe' in methods else (None, None)
    # embedded_lr_support, _ = embedded_log_reg_selector(X, y, num_feats) if 'log-reg' in methods else (None, None)
    # embedded_rf_support, _ = embedded_rf_selector(X, y, num_feats) if 'rf' in methods else (None, None)
    # embedded_lgbm_support, _ = embedded_lgbm_selector(X, y, num_feats) if 'lgbm' in methods else (None, None)

    # # Combine feature lists
    # all_supports = [cor_support, chi_support, rfe_support, embedded_lr_support, embedded_rf_support, embedded_lgbm_support]

    # # Count the votes for each feature
    # votes = sum(map(lambda x: pd.Series(x), all_supports)).fillna(False)
    # votes = votes.applymap(int).sum(axis=1)

    # # Select features with the maximum votes
    # best_features = X.columns[votes == len(all_supports)]
    
    
    #Array Manipulation
    # Find the length of the longest list
    max_length = max(len(cor_feature), len(chi_feature), len(rfe_feature), len(embedded_lr_feature), len(embedded_rf_feature), len(embedded_lgbm_feature))

    # Pad the shorter lists with np.nan and print a message if a list was padded
    def pad_and_warn(feature, name):
        if len(feature) < max_length:
            print(f'############ {name} was padded with np.nan ############')
        return np.pad(feature, (0, max_length - len(feature)), 'constant', constant_values=np.nan)

    cor_feature = pad_and_warn(cor_feature, 'cor_feature')
    chi_feature = pad_and_warn(chi_feature, 'chi_feature')
    rfe_feature = pad_and_warn(rfe_feature, 'rfe_feature')
    embedded_lr_feature = pad_and_warn(embedded_lr_feature, 'embedded_lr_feature')
    embedded_rf_feature = pad_and_warn(embedded_rf_feature, 'embedded_rf_feature')
    embedded_lgbm_feature = pad_and_warn(embedded_lgbm_feature, 'embedded_lgbm_feature')
    
    # cor_feature, chi_feature, rfe_feature, embedded_lr_feature, embedded_rf_feature, embedded_lgbm_feature
    data = {
        'cor_feature': cor_feature,
        'chi_feature': chi_feature,
        'rfe_feature': rfe_feature,
        'embedded_lr_feature': embedded_lr_feature,
        'embedder_rf_feature': embedded_rf_feature,
        'embedded_lgbm_feature': embedded_lgbm_feature
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # If the arrays are of different lengths, you might need to convert them to Series before creating the DataFrame:
    data = {name: pd.Series(arr) for name, arr in data.items()}
    df = pd.DataFrame(data)

    # print(df)

    # Flatten the DataFrame into a single series
    flattened = df.values.flatten()

    # Convert to a pandas Series
    series = pd.Series(flattened)

    # Count the occurrences of each item
    counts = series.value_counts()
    
    # Convert the counts Series to a DataFrame and reset the index
    df_counts = counts.reset_index()

    # Rename the columns
    df_counts.columns = ['Features', 'Total']

    print(df_counts)

    # Print the sum of all counts
    print("Total: " + str(counts.sum()))
    #### Your Code ends here
    return best_features

def preprocess_dataset(dataset_path):
        # Your code starts here (Multiple lines)
        # data - dataset to be analyzed (csv file)
        player_df = pd.read_csv(dataset_path)

        # Chose what columns to remove
        player_df = player_df.drop('Nationality', axis=1)

        # Data manipulation to numeric
        numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
        catcols = ['Preferred Foot','Position','Body Type','Weak Foot']

        # Merging the two datasets
        player_df = player_df[numcols+catcols]

        # Defining traindf and features
        traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
        features = traindf.columns

        # Dropping NaNs
        traindf = traindf.dropna()

        # Creates a new dataframe, and setting the parameter of columns to the features
        traindf = pd.DataFrame(traindf,columns=features)
        
        ### Parameters
        y = traindf['Overall']>=87
        X = traindf.copy()
        del X['Overall']
        
        # The num_feats is the list of numerical features
        num_feats=20
        
        return X, y, num_feats
    #### Your Code ends here
        return best_features

# %%
best_features = autoFeatureSelector(dataset_path="C:\\Users\\HP\\OneDrive\\Desktop\\Applied AI assgn\\ML 1\\fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features

# %% [markdown]
# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features

# %%



