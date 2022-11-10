import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# to evaluate the model
from sklearn.metrics import accuracy_score,roc_auc_score
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Evaluaation function  ########################################################################################################################################################################################################
def eval(y_test, preds):
    return(accuracy_score(y_test, preds), (1 - (accuracy_score(y_test, preds))), roc_auc_score(y_test, preds),   precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds))
    

df = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_1/PE_data.csv')

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
    
    # SVM #################################################################################################################################

    param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear']}  
   
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test) 

    svm_score = np.add(svm_score, eval(y_test, preds))

    
    # LR #################################################################################################################################

    param_grid = {'C': np.logspace(-4, 4, 50),  
              'penalty': ['none', 'l1', 'l2', 'elasticnet'],
              'solver' : ['newton-cg', 'lbfgs', 'liblinear'] 
              }  
   
    grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test) 
    
    LR_score = np.add(LR_score, eval(y_test, preds))
     
    # Naive Bayes ########################################################################################################################

    param_grid = {'var_smoothing': np.logspace(0,-9, num=100)} 
   
    grid = GridSearchCV(GaussianNB(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test) 
    
    NB_score = np.add(NB_score, eval(y_test, preds))
    

       
    # KNN ###########################################################################################################################
   
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
                'metric': ['euclidian', 'manhattan']} 
   
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test)

    KNN_score = np.add(KNN_score, eval(y_test, preds))


    # DT ############################################################################################################################
    param_grid = {'criterion':['gini','entropy'],
                'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
   
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test)
    
    DT_score = np.add(DT_score, eval(y_test, preds))
    
     
    # RF ###############################################################################################################################
    param_grid = { 'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [1,2,3,4,5,6,7,8],
                'criterion' :['gini', 'entropy']
            }
   
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    preds = grid.predict(X_test)

    RF_score = np.add(RF_score, eval(y_test, preds))
     
    break

i = 0

for i in range(len(svm_score)):
    if i == 0: 
        print('Accuracy')    
    elif i == 1:
        print('Error Rate')
    elif i == 2:
        print('AUC')
    elif i == 3:
        print('Precision')
    elif i == 4:
        print('Recall')
    elif i == 5:
        print('F1')
    print(str(svm_score[i]))
    print(str(LR_score[i]))
    print(str(NB_score[i]))
    print(str(KNN_score[i]))
    print(str(DT_score[i]))
    print(str(RF_score[i]))
    
