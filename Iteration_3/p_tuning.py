import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
# to evaluate the model
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluaation function  ########################################################################################################################################################################################################
def eval(y_test, preds):
    return(accuracy_score(y_test, preds), (1 - (accuracy_score(y_test, preds))), roc_auc_score(y_test, preds),   precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds))
    
ss2 = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_2/SS1.csv', index_col=[0])

ss1 = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_2/it1_data_preprocessed.csv', index_col=[0])


ss1 = ss1.drop(['Diabetic', 'Week_1', 'Week_2', 'Week_3', 'Multiple_Fetuses', 'Mass_1', 'Mass_2', 'Mass_3', 'Previous_Stillbirth', 'Previous_Miscarriage', 'Diabetes_Fhistory'], axis=1)
ss2 = ss2.drop(['Week_1', 'Mass_1', 'Previous_Stillbirth', 'Previous_Miscarriage', 'Diabetes_Fhistory'], axis=1)

datasets = [ss1, ss2]

scores = pd.DataFrame(columns = ['Accuracy','Error Rate','ROC AUC','Precision','Recall','F1 Score', 'Best_Params'])

j = 1

for i in datasets:
    df = i
    y = df['Developed_PE']
    df = df.drop('Developed_PE', axis=1)
    # initialise cross-validation

    n_splits = 60

    kf = KFold(n_splits=n_splits, shuffle=False)

    cv = RepeatedStratifiedKFold(n_splits=(10), n_repeats=6, random_state=1)


    # load scaler

    scaler = joblib.load('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/minmax_scaler.gz')


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

        
    X_train, X_test, y_train, y_test  =  train_test_split(df,y)
    

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

    svm_grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    svm_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    svm_preds = svm_grid.predict(X_test)
    svm_score = np.add(svm_score, eval(y_test, svm_preds))

    
    # LR #################################################################################################################################

    param_grid = {'C': np.logspace(-4, 4, 50),  
            'penalty': ['none', 'l1', 'l2', 'elasticnet'],
            'solver' : ['newton-cg', 'lbfgs', 'liblinear'] 
            }  

    lr_grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    lr_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning
    lr_preds = lr_grid.predict(X_test)
    LR_score = np.add(LR_score, eval(y_test, lr_preds))
    
    
    # Naive Bayes ########################################################################################################################

    param_grid = {'var_smoothing': np.logspace(0,-9, num=100)} 

    nb_grid = GridSearchCV(GaussianNB(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    nb_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    preds = nb_grid.predict(X_test)
    NB_score = np.add(NB_score, eval(y_test, preds))
    

    # KNN ###########################################################################################################################

    param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
                'metric': ['euclidian', 'manhattan']} 

    knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    knn_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    knn_preds = knn_grid.predict(X_test)
    KNN_score = np.add(KNN_score, eval(y_test, knn_preds))


    # DT ############################################################################################################################
    param_grid = {'criterion':['gini','entropy'],
                'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

    dt_grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    dt_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    preds = dt_grid.predict(X_test)
    DT_score = np.add(DT_score, eval(y_test, preds))
    
    
    
    # RF ###############################################################################################################################
    param_grid = { 'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [1,2,3,4,5,6,7,8],
                'criterion' :['gini', 'entropy']
            }

    rf_grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    rf_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    rf_preds = rf_grid.predict(X_test)
    RF_score = np.add(RF_score, eval(y_test, rf_preds))
    

    # SGDC ##############################################################################################################################

    param_grid = { 'loss': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['l1', 'l2', 'elasticnet'],
        }

    sgdc_grid = GridSearchCV(SGDClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    sgdc_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning
    preds = sgdc_grid.predict(X_test)
    SGDC_score = np.add(SGDC_score, eval(y_test, preds))

    # GBC #################################################################################################################################

    param_grid = {'learning_rate': [0.01,0.02,0.03],
                'subsample'    : [0.9, 0.5, 0.2],
                'n_estimators' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30],
                'max_depth'    : [4,6,8]
                }

    gbc_grid = GridSearchCV(GradientBoostingClassifier(), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    gbc_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    preds = gbc_grid.predict(X_test)
    GBC_score = np.add(GBC_score, eval(y_test, preds))
    

    # AdaBoost 
    from sklearn.ensemble import AdaBoostClassifier
    
    param_grid = { 
            'algorithm': ['SAMME', 'SAMME.R'],
            'n_estimators':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30],
            'learning_rate':[0.01,0.1]
            }

    ada_grid = GridSearchCV(AdaBoostClassifier(random_state=0), param_grid, refit = True, verbose = 0,n_jobs=-1, cv=cv, scoring = 'recall') 
    
    # fitting the model for grid search 
    ada_grid.fit(X_train, y_train) 
    
    # print best parameter after tuning
    preds = ada_grid.predict(X_test)
    Ada_score = np.add(Ada_score, eval(y_test, preds))

    comb_preds = []
    for i in range(len(lr_preds)):
        x = ((lr_preds[i]) + (svm_preds[i]) + 2*knn_preds[i])/4
        comb_preds.append(round(x))

    comb_score = np.add(comb_score, eval(y_test, comb_preds))

    s1 = {'Model': 'SVM', 'SS': j, 'Accuracy': (svm_score[0]), 'Error Rate': (svm_score[1]), 'ROC AUC': (svm_score[2]), 'Precision': (svm_score[3]), 'Recall': (svm_score[4]), 'F1 Score': (svm_score[5]), 'Best_Params': (svm_grid.best_params_)}
    s2 = {'Model': 'LR', 'SS': j, 'Accuracy': (LR_score[0]), 'Error Rate': (LR_score[1]), 'ROC AUC': (LR_score[2]), 'Precision': (LR_score[3]), 'Recall': (LR_score[4]), 'F1 Score': (LR_score[5]), 'Best_Params': (lr_grid.best_params_)}
    s3 = {'Model': 'NB', 'SS': j, 'Accuracy': (NB_score[0]), 'Error Rate': (NB_score[1]), 'ROC AUC': (NB_score[2]), 'Precision': (NB_score[3]), 'Recall': (NB_score[4]), 'F1 Score': (NB_score[5]), 'Best_Params': (nb_grid.best_params_)}
    s4 = {'Model': 'KNN', 'SS': j, 'Accuracy': (KNN_score[0]), 'Error Rate': (KNN_score[1]), 'ROC AUC': (KNN_score[2]), 'Precision': (KNN_score[3]), 'Recall': (KNN_score[4]), 'F1 Score': (KNN_score[5]), 'Best_Params': (knn_grid.best_params_)}
    s5 = {'Model': 'DT', 'SS': j, 'Accuracy': (DT_score[0]), 'Error Rate': (DT_score[1]), 'ROC AUC': (DT_score[2]), 'Precision': (DT_score[3]), 'Recall': (DT_score[4]), 'F1 Score': (DT_score[5]), 'Best_Params': (dt_grid.best_params_)}
    s6 = {'Model': 'RF', 'SS': j, 'Accuracy': (RF_score[0]), 'Error Rate': (RF_score[1]), 'ROC AUC': (RF_score[2]), 'Precision': (RF_score[3]), 'Recall': (RF_score[4]), 'F1 Score': (RF_score[5]), 'Best_Params': (rf_grid.best_params_)}

    s7 = {'Model': 'SGBC','SS': j,'Accuracy': (SGDC_score[0]), 'Error Rate': (SGDC_score[1]), 'ROC AUC': (SGDC_score[2]), 'Precision': (SGDC_score[3]), 'Recall': (SGDC_score[4]), 'F1 Score': (SGDC_score[5]), 'Best_Params': (sgdc_grid.best_params_)}
    s8 = {'Model': 'GBC','SS': j,'Accuracy': (GBC_score[0]), 'Error Rate': (GBC_score[1]), 'ROC AUC': (GBC_score[2]), 'Precision': (GBC_score[3]), 'Recall': (GBC_score[4]), 'F1 Score': (GBC_score[5]), 'Best_Params': (gbc_grid.best_params_)}
    s9 = {'Model': 'Ada','SS': j,'Accuracy': (Ada_score[0]), 'Error Rate': (Ada_score[1]), 'ROC AUC': (Ada_score[2]), 'Precision': (Ada_score[3]), 'Recall': (Ada_score[4]), 'F1 Score': (Ada_score[5]), 'Best_Params': (ada_grid.best_params_)}

    scores = scores.append([s1,s2,s3,s4,s5,s6,s7, s8, s9], ignore_index=True)
    s10 = {'Model': 'Combined','SS': j,'Accuracy': (comb_score[0]), 'Error Rate': (comb_score[1]), 'ROC AUC': (comb_score[2]), 'Precision': (comb_score[3]), 'Recall': (comb_score[4]), 'F1 Score': (comb_score[5])}
    scores = scores.append([s10], ignore_index=True)

    j += 1


scores.to_csv('./Iteration_3/test6.csv')