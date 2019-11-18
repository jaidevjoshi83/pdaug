import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys,os
from scipy import interp
import pylab as pl
import pandas as pd

###############################################################
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
###############################################################
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
###############################################################
from itertools import cycle
import matplotlib.image as mpimg
from plot_grid_search_digits_final_1 import * 
################################################################

"""
LR = LogisticRegression(solver='lbfgs')
GNB = GaussianNB()
KNB = KNeighborsClassifier()
DT = DecisionTreeClassifier()
SV = SVC(probability=True,gamma='scale')
RF = RandomForestClassifier(n_estimators=10)
SGDC =  SGDClassifier(loss='log')
GBC = GradientBoostingClassifier()
"""

def Data_Label_Gen(InFile):

    df = pd.read_csv(InFile,sep='\t',error_bad_lines=False)
    clm_list = df.columns.tolist()
    X_data = df[clm_list[0:len(clm_list)-1]].values
    y_data = df[clm_list[len(clm_list)-1]].values

    return X_data, y_label

def Fit_Model(InFile, Test_Method, OutDir, OutFile, NoOfFolds):

    if Test_Method == 'Internal':

        for i, (train, test) in enumerate(folds.split(X, y)):

    ############Changes###############################

            if Selected_Sclaer=='Min_Max':
                scaler = MinMaxScaler().fit(X[train])
                x_train = scaler.transform(X[train])
                x_test = scaler.transform(X[test])

            elif Selected_Sclaer=='Standard_Scaler':
                scaler = preprocessing.StandardScaler().fit(X[train])
                x_train = scaler.transform(X[train])
                x_test = scaler.transform(X[test]) 

            elif Selected_Sclaer == 'No_Scaler':
                x_train = X[train]
                x_test = X[test]

            else:
                print('Scalling Method option was not correctly selected...!')


            prob = model.fit(x_train, y[train]).predict_proba(x_test)
            predicted = model.fit(x_train, y[train]).predict(x_test)

            fpr, tpr, thresholds = roc_curve(y[test], prob[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

            TN, FP, FN, TP = confusion_matrix(y[test], predicted).ravel()

            accuracy_score_l.append(round(accuracy_score(y[test], predicted),3))
            a = precision_recall_fscore_support(y[test], predicted, average='macro')
            precision_l.append(round(a[0],3))
            recall_l.append(round(a[1],3))
            f_score_l .append(round(a[2],3))

        accuracy_score_mean = round(float(sum(accuracy_score_l)/float(len(accuracy_score_l))),3)
        precision_mean = round(float(sum(precision_l)/float(len(precision_l))),3)
        recall_mean = round(float(sum(recall_l)/float(len(recall_l))),3)
        f_score_mean = round(float(sum(f_score_l )/float(len(f_score_l ))),3)

        pl.plot([0, 1], [0, 1], '--', lw=2)
        mean_tpr /= folds.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        ##################### Changed #########################################
        V_header = ["accuracy","presision","recall","f1","mean_auc"]
        v_values = [accuracy_score_mean,precision_mean,recall_mean,f_score_mean,mean_auc]
        mname  = ("Logistic_Regression","GaussianNB","KNeighbors","DecisionTree","SVC", "Ranodm Forest","SGDClassifier","GradBoost" )
        #########################################################################

    elif Test_Method == 'External':

            if Selected_Sclaer=='Min_Max':
                scaler = MinMaxScaler().fit(X_train) 
                x_train = scaler.transform(X_train)  
                x_test = scaler.transform(X_test) 
                
            elif Selected_Sclaer=='Standard_Scaler':
                scaler = preprocessing.StandardScaler().fit(X_train)
                x_train = scaler.transform(X_train)
                x_test = scaler.transform(X_test) 

            elif Selected_Sclaer == 'No_Scaler':
                x_train = X_train
                x_test = X_test

            else:
                print('Scalling Method option was not correctly selected...!')


            prob = model.fit(x_train, y_train).predict_proba(x_test)
            predicted = model.fit(x_train, y_train).predict(x_test)

            fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

            TN, FP, FN, TP = confusion_matrix(y_test, predicted).ravel()

            accuracy_score_mean = round(accuracy_score(y_test, predicted),3)
            a = precision_recall_fscore_support(y_test, predicted, average='macro')
            precision = round(a[0],3)
            recall= round(a[1],3)
            f_score= round(a[2],3)

            pl.plot([0, 1], [0, 1], '--', lw=2)
            mean_tpr /= folds.get_n_splits(X, y)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)

            ##################### Changed #########################################
            V_header = ["accuracy","presision","recall","f1","mean_auc"]
            v_values = [accuracy_score_mean,precision_mean,recall_mean,f_score_mean,mean_auc]
            mname  = ("Logistic_Regression","GaussianNB","KNeighbors","DecisionTree","SVC", "Ranodm Forest","SGDClassifier","GradBoost" )
            

    return V_header, v_values, mean_fpr, mean_tpr, mean_auc

def SVM_classification(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, random_state):

    pera = {"C":C, 
    "kernel":kernel, 
    "degree":degree, 
    "gamma":gamma, 
    "coef0":coef0, 
    "shrinking":shrinking,
    "probability":probability, 
    "tol":tol, 
    "cache_size":cache_size, 
    "class_weight":class_weight, 
    "verbose":verbose, 
    "max_iter":max_iter, 
    "decision_function_shape":decision_function_shape, 
    "random_state":random_state}

    model = SVC(pera**)

    

    prob = model.fit(X[train], y[train]).predict_proba(X[test])
    predicted = model.fit(X[train], y[train]).predict(X[test])

def SGD_Classification( loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon, n_jobs,
    random_state, learning_rate, eta0, power_t, early_stopping, validation_fraction, n_iter_no_change, class_weight, warm_start, average):

    pera = {"loss":loss, 
    "penalty":penalty, 
    "alpha":alpha,
    "l1_ratio":l1_ratio,
    "fit_intercept":fit_intercept, 
    "max_iter":max_iter, 
    "tol":tol, 
    "shuffle":shuffle, 
    "verbose":verbose, 
    "epsilon":epsilon, 
    "n_jobs":n_jobs, 
    "random_state":random_state, 
    "learning_rate":learning_rate, 
    "eta0":eta0, 
    "power_t":power_t, 
    "early_stopping":early_stopping, 
    "validation_fraction":validation_fraction, 
    "n_iter_no_change":n_iter_no_change, 
    "class_weight":class_weight, 
    "warm_start":warm_start, 
    "average":average}

def DecisionTree_Classification(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, 
    random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, presort):

    pera = {"criterion":criterion,
    "splitter":splitter, 
    "max_depth":max_depth, 
    "min_samples_split":min_samples_split, 
    "min_samples_leaf":min_samples_leaf, 
    "min_weight_fraction_leaf":min_weight_fraction_leaf, 
    "max_features":max_features, 
    "random_state":random_state, 
    "max_leaf_nodes":max_leaf_nodes, 
    "min_impurity_decrease":min_impurity_decrease, 
    "min_impurity_split":min_impurity_split, 
    "class_weight":class_weight, 
    "presort":presort}:

def GradientBoosting_Classification(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
    max_depth, min_impurity_decrease,min_impurity_split, init, random_state, max_features, verbose, max_leaf_nodes, warm_start, presort, validation_fraction, n_iter_no_change, tol):

    pera = {"loss":loss, 
    "learning_rate":learning_rate, 
    "n_estimators":n_estimators, 
    "subsample":subsample, 
    "criterion":criterion, 
    "min_samples_split":min_samples_split, 
    "min_samples_leaf":min_samples_leaf, 
    "min_weight_fraction_leaf":min_weight_fraction_leaf, 
    "max_depth":max_depth, 
    "min_impurity_decrease":min_impurity_decrease,
    "min_impurity_split":min_impurity_split, 
    "init":init, 
    "random_state":random_state, 
    "max_features":max_features, 
    "verbose":verbose, 
    "max_leaf_nodes":max_leaf_nodes, 
    "warm_start":warm_start, 
    "presort":presort, 
    "validation_fraction":validation_fraction, 
    "n_iter_no_change":n_iter_no_change, 
    "tol":tol}:

def RandomForestClassifier( n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, 
    min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, class_weight):

    pera = {"n_estimators":n_estimators, 
    "criterion":criterion, 
    "max_depth":max_depth, 
    "min_samples_split":min_samples_split, 
    "min_samples_leaf":min_samples_leaf, 
    "min_weight_fraction_leaf":min_weight_fraction_leaf, 
    "max_features":max_features, 
    "max_leaf_nodes":max_leaf_nodes, 
    "min_impurity_decrease":min_impurity_decrease, 
    "min_impurity_split":min_samples_split, 
    "bootstrap":bootstrap, 
    "oob_score":oob_score, 
    "n_jobs":n_jobs, 
    "random_state":random_state, 
    "verbose":verbose, 
    "warm_start":warm_start, 
    "class_weight":class_weight}

def LogisticRegression(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start n_jobs, l1_ratio):

    pera = {"penalty":penalty, 
    "dual":dual, 
    "tol":tol, 
    "C":C, 
    "fit_intercept":fit_intercept, 
    "intercept_scaling":intercept_scaling, 
    "class_weight":class_weight, 
    "random_state":random_state, 
    "solver":solver, 
    "max_iter":max_iter, 
    "multi_class":multi_class, 
    "verbose":verbose, 
    "warm_start":warm_start
    "n_jobs":n_jobs, 
    "l1_ratio":l1_ratio}

def KNeighbors_Classifier(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params,  n_jobs):

    pera = {"weights":weights, "algorithm":algorithm, "leaf_size"leaf_size, "p":p, "metric":metric, "metric_params":metric_params, "n_jobs":n_jobs}

    return 

def GaussianNB(  priors, var_smoothing):  

    pera = {"priors":priors, 
    "var_smoothing"var_smoothing}:     


def main(algo, InFile)

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    svmc = subparsers.add_parser('SVMC')
    svmc.add_argument("--InData",required=True,default=None, help="")
    svmc.add_argument("--C", required=False, default=1.0, help="")
    svmc.add_argument("--kernel", required=False, default='rbf', help="")
    svmc.add_argument("--degree", required=False, default=3, help="")
    svmc.add_argument("--gamma", required=False, default='auto_deprecated', help="")
    svmc.add_argument("--coef0", required=False, default=0.0, help="")
    svmc.add_argument("--shrinking", required=False, default='True', help="")
    svmc.add_argument("--probability", required=False, default='False', help="")
    svmc.add_argument("--tol", required=False, default=0.001, help="")
    svmc.add_argument("--cache_size", required=False, default=200, help="")
    svmc.add_argument("--class_weight", required=False, default='False', help="")
    svmc.add_argument("--verbose", required=False, default='False', help="")
    svmc.add_argument("--max_iter", required=False, default=-1, help="")
    svmc.add_argument("--decision_function_shape", required=False, default='ovr', help="")
    svmc.add_argument("--random_state", required=False, default='None', help="")
    svmc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    sgdc = subparsers.add_parser('SGDC')
    sgdc.add_argument("--InData",required=True, default=None, help="")
    sgdc.add_argument("--loss", required=False, default='hinge', help="")
    sgdc.add_argument("--penalty", required=False, default='l2', help="")
    sgdc.add_argument("--alpha", required=False, default=0.0001, help="")
    sgdc.add_argument("--l1_ratio", required=False, default=0.15, help="")
    sgdc.add_argument("--fit_intercept", required=False, default='True', help="")
    sgdc.add_argument("--max_iter", required=False, default=1000, help="")
    sgdc.add_argument("--tol", required=False, default=0.001, help="")
    sgdc.add_argument("--shuffle", required=False, default='True', help="")
    sgdc.add_argument("--verbose", required=False, default=0, help="")
    sgdc.add_argument("--epsilon", required=False, default=0.1, help="")
    sgdc.add_argument("--n_jobs", required=False, default='None', help="")
    sgdc.add_argument("--random_state", required=False, default='None', help="")
    sgdc.add_argument("--learning_rate", required=False, default='optimal', help="")
    sgdc.add_argument("--eta0", required=False, default=0.0, help="")
    sgdc.add_argument("--power_t", required=False, default=0.5, help="")
    sgdc.add_argument("--early_stopping", required=True, default='False', help="MinMaxScaler")
    sgdc.add_argument("--validation_fraction", required=False, default=0.1, help="")
    sgdc.add_argument("--n_iter_no_change", required=False, default=5, help="")
    sgdc.add_argument("--class_weight", required=False, default='None', help="")
    sgdc.add_argument("--warm_start", required=False, default='False', help="")
    sgdc.add_argument("--average", required=True, default='False', help="MinMaxScaler")
    sgdc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    dtc = subparsers.add_parser('DTC')
    dtc.add_argument("--InData",required=True, default=None, help="")
    dtc.add_argument("--criterion", required=False, default='gini', help="")
    dtc.add_argument("--splitter", required=False, default='best', help="" )
    dtc.add_argument("--max_depth", required=False, default='None', help="")
    dtc.add_argument("--min_samples_split", required=False, default=2, help="")
    dtc.add_argument("--min_samples_leaf", required=False, default=1, help="")
    dtc.add_argument("--min_weight_fraction_leaf", required=False, default=0.0, help="")
    dtc.add_argument("--max_features", required=False, default='None', help="")
    dtc.add_argument("--random_state", required=False, default='None', help="")
    dtc.add_argument("--max_leaf_nodes", required=False, default='None', help="")
    dtc.add_argument("--min_impurity_decrease", required=False, default=0.0, help="")
    dtc.add_argument("--min_impurity_split", required=False, default='None', help="")
    dtc.add_argument("--class_weight", required=False, default='None', help="")
    dtc.add_argument("--presort=False", required=False, default='False', help="")
    dtc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    gbc =  subparsers.add_parser('GBC')
    gbc.add_argument("--InData", required=True, default=None, help="" )
    gbc.add_argument("--loss", required=False, default='deviance', 
    gbc.add_argument("--learning_rate", required=False, default=0.1, 
    gbc.add_argument("--n_estimators", required=False, default=100, 
    gbc.add_argument("--subsample", required=False, default=1.0, 
    gbc.add_argument("--criterion", required=False,default='friedman_mse', 
    gbc.add_argument("--min_samples_split", required=False, default=2, 
    gbc.add_argument("--min_samples_leaf", required=False, default=1, 
    gbc.add_argument("--min_weight_fraction_leaf", required=False, default=0.0, 
    gbc.add_argument("--max_depth", required=False, default=3, 
    gbc.add_argument("--min_impurity_decrease", required=False, default=0.0,
    gbc.add_argument("--min_impurity_split", required=False, default='None', 
    gbc.add_argument("--init", required=False,default='None', 
    gbc.add_argument("--random_state", required=False, default='None', 
    gbc.add_argument("--max_features", required=False, default='None', 
    gbc.add_argument("--verbose",required=False,default=0, 
    gbc.add_argument("--max_leaf_nodes", required=False, default='None', 
    gbc.add_argument("--warm_start", required=False, default='False', 
    gbc.add_argument("--presort", required=False,default='auto', 
    gbc.add_argument("--validation_fraction", required=False,default=0.1, 
    gbc.add_argument("--n_iter_no_change", required=False, default='None', 
    gbc.add_argument("--tol", required=False, default=0.0001,
    gbc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    rfc =  subparsers.add_parser('RFC')
    tfc.add_argument("--InData", required=True, default=None, help="" )
    tfc.add_argument("--n_estimators", required=False, default='warn', 
    tfc.add_argument("--criterion", required=False,default='gini', 
    tfc.add_argument("--max_depth", required=False,default='None', 
    tfc.add_argument("--min_samples_split", required=False,default=2, 
    tfc.add_argument("--min_samples_leaf", required=False,default=1, 
    tfc.add_argument("--min_weight_fraction_leaf", required=False,default=0.0 )
    tfc.add_argument("--max_features", required=False, default='auto',) 
    tfc.add_argument("--max_leaf_nodes", required=False, default='None', )
    tfc.add_argument("--min_impurity_decrease", required=False, default=0.0,) 
    tfc.add_argument("--min_impurity_split", required=False, default='None',) 
    tfc.add_argument("--bootstrap", required=False,default='True', )
    tfc.add_argument("--oob_score", required=False, default='False', )
    tfc.add_argument("--n_jobs", required=False, default='None', )
    tfc.add_argument("--random_state", required=False,default='None',) 
    tfc.add_argument("--verbose", required=False, default=0, )
    tfc.add_argument("--warm_start", required=False, default='False',) 
    tfc.add_argument("--class_weight", required=False, default='None',)
    tfc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    lrc =  subparsers.add_parser('LRC')
    lrc.add_argument("--InData", required=True, default=None, help="" )
    lrc.add_argument("--penalty", default='l2', )
    lrc.add_argument("--dual", default='False', )
    lrc.add_argument("--tol", default=0.0001, )
    lrc.add_argument("--C", default=1.0, )
    lrc.add_argument("--fit_intercept", default='True', )
    lrc.add_argument("--intercept_scaling", default=1, )
    lrc.add_argument("--class_weight", default='None')
    lrc.add_argument("--random_state", default='None',) 
    lrc.add_argument("--solver", default='warn') 
    lrc.add_argument("--max_iter", default=100), 
    lrc.add_argument("--multi_class", default='warn') 
    lrc.add_argument("--verbose", default=0, )
    lrc.add_argument("--warm_start", default='False', )
    lrc.add_argument("--n_jobs", default='None', )
    lrc.add_argument("--l1_ratio", default='None',)
    tfc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    knc = subparsers.add_parser('KNC')
    knc.add_argument("--InData", required=True, default=None, help="" )
    knc.add_argument("--n_neighbors", required=False, default=5,)
    knc.add_argument("--weights",required=False, default='uniform',) 
    knc.add_argument("--algorithm", required=False, default='auto', )
    knc.add_argument("--leaf_size", required=False, default=30, )
    knc.add_argument("--p", required=False, default=2, )
    knc.add_argument("--metric", required=False, default='minkowski', )
    knc.add_argument("--metric_params", required=False, default='None' )
    knc.add_argument("--n_jobs", required=False, default='None',)
    knc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    gnbc = subparsers.add_parser('GNBC')
    gnbc.add_argument("--InData", required=True, default=None, help="" )
    gnbc.add_argument("--priors", required=False, default='None',)
    gnbc.add_argument("--var_smoothing", required=False, default=1e-09,)
    gnbc.add_argument("--OutDir", required=True, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")

    args = parser.parse_args()
