import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# to evaluate the model
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Evaluaation function  ########################################################################################################################################################################################################
def eval(y_test, preds):
    return(accuracy_score(y_test, preds), (1 - (accuracy_score(y_test, preds))), roc_auc_score(y_test, preds),   precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds))
    

df = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_2/it1_data_preprocessed.csv', index_col=[0])

y = df['Developed_PE']
df = df.drop('Developed_PE', axis=1)


# initialise cross-validation

n_splits = 60

kf = KFold(n_splits=n_splits, shuffle=False)

cv = RepeatedStratifiedKFold(n_splits=(10), n_repeats=6, random_state=1)


# load scaler

scaler = joblib.load('C:/Users/no1ca/OneDrive/Documents/Masters/masters_PE_model/minmax_scaler.joblib')


# arrays to store scores from CV

svm_score = [0, 0, 0, 0, 0, 0]
LR_score = [0, 0, 0, 0, 0, 0]
NB_score = [0, 0, 0, 0, 0, 0]
SGDC_score = [0, 0, 0, 0, 0, 0]
KNN_score = [0, 0, 0, 0, 0, 0]
DT_score = [0, 0, 0, 0, 0, 0]
RF_score = [0, 0, 0, 0, 0, 0]
GBC_score = [0, 0, 0, 0, 0, 0]
Ada_score = [0, 0, 0, 0, 0, 0]
comb_score = [0, 0, 0, 0, 0, 0]


# loop and split for each split

for train_index, test_index in cv.split(df, y):
    
    X_train =  df.iloc[train_index]
    X_test =  df.iloc[test_index]
    y_train  =  y.iloc[train_index]
    y_test =  y.iloc[test_index]
    

    # scale features using min and max    
    #  fit  the scaler to the train set

    scaler.fit(X_train) 
    

    # transform the train and test set
    
    X_train = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns
    )
    
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns
    )
    
    
    # Models ##############################################################################################################################
    
    
    # SGDC ##############################################################################################################################

    param_grid = { 'loss': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'penalty': ['l1', 'l2', 'elasticnet'],
            }
   
    grid = GridSearchCV(SGDClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test)
    SGDC_score = np.add(SGDC_score, eval(y_test, preds))

    # GBC #################################################################################################################################

    param_grid = {'learning_rate': [0.01,0.02,0.03],
                  'subsample'    : [0.9, 0.5, 0.2],
                  'n_estimators' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30],
                  'max_depth'    : [4,6,8]
                 }
   
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test)
    GBC_score = np.add(GBC_score, eval(y_test, preds))
    
    
    # AdaBoost 
    from sklearn.ensemble import AdaBoostClassifier
    
    param_grid = { 
              'algorithm': ['SAMME', 'SAMME.R'],
              'n_estimators':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30],
              'learning_rate':[0.01,0.1]
              }
   
    grid = GridSearchCV(AdaBoostClassifier(random_state=0), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test)
    Ada_score = np.add(Ada_score, eval(y_test, preds))
    
    break

    
