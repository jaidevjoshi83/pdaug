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
#from plot_grid_search_digits_final_1 import * 
################################################################

def ReturnData(TrainFile,  TestMethod, TestFile=None):
    
    if (TestFile == None) and (TestMethod == 'Internal' or 'CrossVal'):

        df = pd.read_csv(TrainFile, sep='\t')
        clm_list = df.columns.tolist()
        X_train = df[clm_list[0:len(clm_list)-1]].values
        y_train = df[clm_list[len(clm_list)-1]].values
        X_test = None 
        y_test = None

        return X_train, y_train, X_test, y_test

    elif (TestFile is not None) and (TestMethod == 'External'):

        df = pd.read_csv(TrainFile, sep='\t')
        clm_list = df.columns.tolist()
        X_train = df[clm_list[0:len(clm_list)-1]].values
        y_train = df[clm_list[len(clm_list)-1]].values
        df1 = pd.read_csv(TestFile, sep='\t')
        clm_list = df1.columns.tolist()
        X_test = df1[clm_list[0:len(clm_list)-1]].values
        y_test = df1[clm_list[len(clm_list)-1]].values

        return X_train, y_train, X_test, y_test

    elif (TestFile is not None) and (TestMethod == 'Predict'):

        df = pd.read_csv(TrainFile, sep='\t')
        clm_list = df.columns.tolist()
        X_train = df[clm_list[0:len(clm_list)-1]].values
        y_train = df[clm_list[len(clm_list)-1]].values

        df = pd.read_csv(TestFile, sep='\t')
        X_test = df
        y_test = None

        return X_train, y_train, X_train, y_train

def Fit_Model(TrainData, Test_Method, Algo, Selected_Sclaer, NoOfFolds=None, TestSize=None, TestData=None ):

    if Test_Method == 'Internal':

        X,y,_,_ = ReturnData(TrainData, Test_Method)

        #print X,y

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        specificity_list = []
        sensitivity_list = []
        presison_list = []
        mcc_list =  []
        f1_list = []
        
        folds = StratifiedKFold(n_splits=5)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        ######################################################
        accuracy_score_l = []
        cohen_kappa_score_l = []
        matthews_corrcoef_l = []
        precision_l = []
        recall_l = []
        f_score_l = []
        #####################################

        folds = StratifiedKFold(n_splits=5)

        for i, (train, test) in enumerate(folds.split(X, y)):

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
            #print x_train, y[train],x_test
            print (x_test)

            prob = Algo.fit(x_train, y[train]).predict_proba(x_test)
            predicted = Algo.fit(x_train, y[train]).predict(x_test)

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

        pl.figure()
        #pl.plot([0, 1], [0, 1],'--', lw=2)
        mean_tpr /= folds.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        pl.plot(mean_fpr, mean_tpr, '-', color='red',label='AUC = %0.2f' % mean_auc, lw=2)

        #pl.plot([0, 1], [0, 1], 'k--', lw=lw)
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC Cureve for All the classifier')
        pl.legend(loc="lower right")
        pl.savefig('ROC.png') 
        #pl.show()

        ###############################################################################################################################
        V_header = ["accuracy","presision","recall","f1","mean_auc"]                                                                  #
        v_values = [accuracy_score_mean,precision_mean,recall_mean,f_score_mean,mean_auc]                                             # 
        mname  = ("Logistic_Regression","GaussianNB","KNeighbors","DecisionTree","SVC", "Ranodm Forest","SGDClassifier","GradBoost" ) #
        ###############################################################################################################################

        return  V_header, v_values

    elif Test_Method == 'External':

        X_train,y_train,X_test,y_test = ReturnData(TrainData, Test_Method, TestData)

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
        TN, FP, FN, TP = confusion_matrix(y_test, predicted).ravel()
        accu_score = accuracy_score(y_test, predicted)

        a = precision_recall_fscore_support(y_test, predicted, average='macro')

        pre_score = round(a[0],3)
        recall_score= round(a[1],3)
        f_score= round(a[2],3)

        pl.plot(fpr, tpr, '--', lw=2)
        auc_score = auc(fpr, tpr)

        a = precision_recall_fscore_support(y_test, predicted, average='macro')
        pre_score = round(a[0],3)
        rec_score = round(a[1],3)
        f_score = round(a[2],3)

        V_header = ["accuracy","presision","recall","f1","mean_auc"]
        v_values = [accu_score, pre_score, rec_score, f_score, auc_score]

        #print v_values

        return V_header, v_values

    elif Test_Method == "TestSplit":

        X_train,y_train,_,_ = ReturnData(TrainData, Test_Method)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TestSize, random_state=0)

        #print X_train.shape, y_train.shape, X_test.shape,  y_test.shape 

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
        accu_score = accuracy_score(y_test, predicted)

        a = precision_recall_fscore_support(y_test, predicted, average='macro')

        pre_score = round(a[0],3)
        recall_score= round(a[1],3)
        f_score= round(a[2],3)

        pl.plot(fpr, tpr, '-', color='red',label='AUC = %0.2f' % accu_score, lw=2)

        #pl.plot([0, 1], [0, 1], 'k--', lw=lw)
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC Cureve for All the classifier')
        pl.legend(loc="lower right")
        pl.savefig('ROC.png') 
        pl.plot(fpr, tpr, '--', lw=2)
        
        auc_score = auc(fpr, tpr)

        a = precision_recall_fscore_support(y_test, predicted, average='macro')
        pre_score = round(a[0],3)
        rec_score = round(a[1],3)
        f_score = round(a[2],3)

        V_header = ["accuracy","presision","recall","f1","mean_auc"]
        v_values = [accu_score, pre_score, rec_score, f_score, auc_score]
 
        return v_values

    elif Test_Method == "Predict":

        X_train, y_train, X_test, _ = ReturnData(TrainData, Test_Method,TestData)

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

        predicted = model.fit(x_train, y_train).predict(x_test)

        return predicted

def SVM_Classifier(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, randomRtate, TrainFile, TestMethod,  SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):
   
    pera={ 

    'C':C, 
    'kernel':kernel, 
    'degree':degree, 
    'gamma':gamma, 
    'coef0':coef0, 
    'shrinking':shrinking, 
    'probability':True,
    'tol':tol, 
    'cache_size':cache_size, 
    'class_weight':class_weight, 
    'verbose':verbose, 
    'max_iter':max_iter, 
    'decision_function_shape':decision_function_shape, 
    'probability':True, 
    'random_state':int(randomRtate)
    }

    model = SVC( **pera )
    print (model)

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def SGD_Classifier( loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon, n_jobs,
    random_state, learning_rate, eta0, power_t, early_stopping, validation_fraction, n_iter_no_change, class_weight, warm_start, average, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):

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

    model =  SGDClassifier(**pera)

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def DT_Classifier(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,  random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, presort, ccpalpha, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):

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
    "presort":presort,
    "ccp_alpha":ccpalpha}

    model = DecisionTreeClassifier(**pera)
    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def GB_Classifier(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
    max_depth, min_impurity_decrease,min_impurity_split, init, random_state, max_features, verbose, max_leaf_nodes, warm_start, presort, validation_fraction, n_iter_no_change, tol, ccpalpha, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):

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
    "tol":tol,
    "ccp_alpha":ccpalpha}

    model =  GradientBoostingClassifier(**pera)
    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def RF_Classifier( n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease,  min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, class_weight, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):


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

    model = RandomForestClassifier(**pera)
    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def LR_Classifier(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):

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
    "warm_start":warm_start,
    "n_jobs":n_jobs, 
    "l1_ratio":l1_ratio}

    model = LogisticRegression(**pera)
    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def KN_Classifier(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params,  n_jobs, TrainFile, TestMethod,  SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):

    pera = {"weights":weights, 
    "algorithm":algorithm, 
    "leaf_size":leaf_size, 
    "p":p, "metric":metric, 
    "metric_params":metric_params, 
    "n_jobs":n_jobs}
    
    model = KNeighborsClassifier(**pera)
    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def GNB_Classifier( priors, var_smoothing, TrainFile, TestMethod,  SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, OfileName, Workdirpath):  

    pera = {"priors":priors, 
    "var_smoothing":var_smoothing}  

    model = GaussianNB(**pera) 
    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile )
  

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    svmc = subparsers.add_parser('SVMC')
    svmc.add_argument("--C", required=False, default=1.0, help="")
    svmc.add_argument("--kernel", required=False, default='rbf', help="")
    svmc.add_argument("--degree", required=False, default=3, help="")
    svmc.add_argument("--gamma", required=False, default='auto_deprecated', help="")
    svmc.add_argument("--coef0", required=False, default=0.0, help="")
    svmc.add_argument("--shrinking", required=False, default=True, help="")
    svmc.add_argument("--probability", required=False, default=False, help="")
    svmc.add_argument("--tol", required=False, default=0.001, help="")
    svmc.add_argument("--cache_size", required=False, default=200, help="")
    svmc.add_argument("--class_weight", required=False, default=None, help="")
    svmc.add_argument("--verbose", required=False, default=False, help="")
    svmc.add_argument("--max_iter", required=False, default=-1, help="")
    svmc.add_argument("--decision_function_shape", required=False, default='ovr', help="")
    svmc.add_argument("--randomState", required=False, default=None) 
    svmc.add_argument("--TrainFile", required=True, default=None, help="")
    svmc.add_argument("--TestMethod", required=True, default=None, help="")
    svmc.add_argument("--SelectedSclaer", required=True, help="")
    svmc.add_argument("--NFolds", required=False, default=5, help="")
    svmc.add_argument("--Testspt", required=False, default=0.2, help="")
    svmc.add_argument("--TestFile", required=False, default=None, help="")
    svmc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    svmc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    svmc.add_argument("--htmlFname", required=False, help="")
    svmc.add_argument("--OfileName", required=False, help="")
    svmc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    sgdc = subparsers.add_parser('SGDC')
    sgdc.add_argument("--loss", required=False, default='log', help="")
    sgdc.add_argument("--penalty", required=False, default='l2', help="")
    sgdc.add_argument("--alpha", required=False, default=0.0001, help="")
    sgdc.add_argument("--l1_ratio", required=False, default=0.15, help="")
    sgdc.add_argument("--fit_intercept", required=False, default=True, help="")
    sgdc.add_argument("--max_iter", required=False, default=1000, help="")
    sgdc.add_argument("--tol", required=False, default=0.001, help="")
    sgdc.add_argument("--shuffle", required=False, default=True, help="")
    sgdc.add_argument("--verbose", required=False, default=0, help="")
    sgdc.add_argument("--epsilon", required=False, default=0.1, help="")
    sgdc.add_argument("--n_jobs", required=False, default=None, help="")
    sgdc.add_argument("--random_state", required=False, default=None, help="")
    sgdc.add_argument("--learning_rate", required=False, default='optimal', help="")
    sgdc.add_argument("--eta0", required=False, default=0.0, help="")
    sgdc.add_argument("--power_t", required=False, default=0.5, help="")
    sgdc.add_argument("--early_stopping", required=False, default=False, help="MinMaxScaler")
    sgdc.add_argument("--validation_fraction", required=False, default=0.1, help="")
    sgdc.add_argument("--n_iter_no_change", required=False, default=5, help="")
    sgdc.add_argument("--class_weight", required=False, default=None, help="")
    sgdc.add_argument("--warm_start", required=False, default=False, help="")
    sgdc.add_argument("--average", required=False, default=False, help="MinMaxScaler")
    sgdc.add_argument("--TrainFile", required=True, default=None, help="")
    sgdc.add_argument("--TestMethod", required=True, default=None, help="")
    sgdc.add_argument("--SelectedSclaer", required=True, help="")
    sgdc.add_argument("--NFolds", required=False, default=5, help="")
    sgdc.add_argument("--Testspt", required=False, default=0.2, help="")
    sgdc.add_argument("--TestFile", required=False, default=None, help="")
    sgdc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    sgdc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    sgdc.add_argument("--htmlFname", required=False, help="")
    sgdc.add_argument("--OfileName", required=False, help="")
    sgdc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    dtc = subparsers.add_parser('DTC') 
    dtc.add_argument("--criterion", required=False, default='gini', help="")
    dtc.add_argument("--splitter", required=False, default='best', help="" )
    dtc.add_argument("--max_depth", required=False, default=None, help="")
    dtc.add_argument("--min_samples_split", required=False, default=2, help="")
    dtc.add_argument("--min_samples_leaf", required=False, default=1, help="")
    dtc.add_argument("--min_weight_fraction_leaf", required=False, default=0.0, help="")
    dtc.add_argument("--max_features", required=False, default=None, help="")
    dtc.add_argument("--random_state", required=False, default=None, help="")
    dtc.add_argument("--max_leaf_nodes", required=False, default=None, help="")
    dtc.add_argument("--min_impurity_decrease", required=False, default=0.0, help="")
    dtc.add_argument("--min_impurity_split", required=False, default=None, help="")
    dtc.add_argument("--class_weight", required=False, default=None, help="")
    dtc.add_argument("--presort", required=False, default=False, help="")
    dtc.add_argument("--ccpalpha", required=False, default=0.0, help="")
    dtc.add_argument("--TrainFile", required=True, default=None, help="")
    dtc.add_argument("--TestMethod", required=True, default=None, help="")
    dtc.add_argument("--SelectedSclaer", required=True, help="")
    dtc.add_argument("--NFolds", required=False, default=5, help="")
    dtc.add_argument("--Testspt", required=False, default=0.2, help="")
    dtc.add_argument("--TestFile", required=False, default=None, help="")
    dtc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    dtc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    dtc.add_argument("--htmlFname", required=False, help="")
    dtc.add_argument("--OfileName", required=False, help="")
    dtc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    gbc =  subparsers.add_parser('GBC')
    gbc.add_argument("--loss", required=False, default='deviance') 
    gbc.add_argument("--learning_rate", required=False, default=0.1) 
    gbc.add_argument("--n_estimators", required=False, default=100) 
    gbc.add_argument("--subsample", required=False, default=1.0) 
    gbc.add_argument("--criterion", required=False,default='friedman_mse') 
    gbc.add_argument("--min_samples_split", required=False, default=2) 
    gbc.add_argument("--min_samples_leaf", required=False, default=1) 
    gbc.add_argument("--min_weight_fraction_leaf", required=False, default=0.0) 
    gbc.add_argument("--max_depth", required=False, default=3) 
    gbc.add_argument("--min_impurity_decrease", required=False, default=0.0)
    gbc.add_argument("--min_impurity_split", required=False, default=None) 
    gbc.add_argument("--init", required=False,default=None) 
    gbc.add_argument("--random_state", required=False, default=None) 
    gbc.add_argument("--max_features", required=False, default=None) 
    gbc.add_argument("--verbose",required=False,default=0) 
    gbc.add_argument("--max_leaf_nodes", required=False, default=None) 
    gbc.add_argument("--warm_start", required=False, default=False) 
    gbc.add_argument("--presort", required=False,default='auto') 
    gbc.add_argument("--validation_fraction", required=False,default=0.1) 
    gbc.add_argument("--n_iter_no_change", required=False, default=None) 
    gbc.add_argument("--tol", required=False, default=0.0001)
    gbc.add_argument("--ccpalpha", required=False, default=0.0, help="")
    gbc.add_argument("--TrainFile", required=True, default=None, help="")
    gbc.add_argument("--TestMethod", required=True, default=None, help="")
    gbc.add_argument("--SelectedSclaer", required=True, help="")
    gbc.add_argument("--NFolds", required=False, default=5, help="")
    gbc.add_argument("--Testspt", required=False, default=0.2, help="")
    gbc.add_argument("--TestFile", required=False, default=None, help="")
    gbc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    gbc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    gbc.add_argument("--htmlFname", required=False, help="")
    gbc.add_argument("--OfileName", required=False, help="")
    gbc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    rfc = subparsers.add_parser('RFC')
    rfc.add_argument("--n_estimators", required=False, default=100) 
    rfc.add_argument("--criterion", required=False,default='gini') 
    rfc.add_argument("--max_depth", required=False,default=None) 
    rfc.add_argument("--min_samples_split", required=False,default=2) 
    rfc.add_argument("--min_samples_leaf", required=False,default=1) 
    rfc.add_argument("--min_weight_fraction_leaf", required=False,default=0.0)
    rfc.add_argument("--max_features", required=False, default='auto') 
    rfc.add_argument("--max_leaf_nodes", required=False, default=None)
    rfc.add_argument("--min_impurity_decrease", required=False, default=0.0) 
    rfc.add_argument("--min_impurity_split", required=False, default=None) 
    rfc.add_argument("--bootstrap", required=False,default=True)
    rfc.add_argument("--oob_score", required=False, default=False)
    rfc.add_argument("--n_jobs", required=False, default=None )
    rfc.add_argument("--random_state", required=False,default=None) 
    rfc.add_argument("--verbose", required=False, default=0, )
    rfc.add_argument("--warm_start", required=False, default=False) 
    rfc.add_argument("--class_weight", required=False, default=None)
    rfc.add_argument("--TrainFile", required=True, default=None, help="")
    rfc.add_argument("--TestMethod", required=True, default=None, help="")
    rfc.add_argument("--SelectedSclaer", required=True, help="")
    rfc.add_argument("--NFolds", required=False, default=5, help="")
    rfc.add_argument("--Testspt", required=False, default=0.2, help="")
    rfc.add_argument("--TestFile", required=False, default=None, help="")
    rfc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    rfc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    rfc.add_argument("--htmlFname", required=False, help="")
    rfc.add_argument("--OfileName", required=False, help="")
    rfc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    lrc =  subparsers.add_parser('LRC')
    lrc.add_argument("--penalty", default='l2', )
    lrc.add_argument("--dual", default=False, )
    lrc.add_argument("--tol", default=0.0001, )
    lrc.add_argument("--C", default=1.0, )
    lrc.add_argument("--fit_intercept", default=True )
    lrc.add_argument("--intercept_scaling", default=1 )
    lrc.add_argument("--class_weight", default=None)
    lrc.add_argument("--random_state", default=None) 
    lrc.add_argument("--solver", required=False, default='lbfgs') 
    lrc.add_argument("--max_iter", default=100), 
    lrc.add_argument("--multi_class", default='auto') 
    lrc.add_argument("--verbose", default=0, )
    lrc.add_argument("--warm_start", default=False, )
    lrc.add_argument("--n_jobs", default=None, )
    lrc.add_argument("--l1_ratio", default=None,)
    lrc.add_argument("--TrainFile", required=True, default=None, help="")
    lrc.add_argument("--TestMethod", required=True, default=None, help="")
    lrc.add_argument("--SelectedSclaer", required=True, help="")
    lrc.add_argument("--NFolds", required=False, default=5, help="")
    lrc.add_argument("--Testspt", required=False, default=0.2, help="")
    lrc.add_argument("--TestFile", required=False, default=None, help="")
    lrc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    lrc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    lrc.add_argument("--htmlFname", required=False, help="")
    lrc.add_argument("--OfileName", required=False, help="")
    lrc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    knc = subparsers.add_parser('KNC')
    knc.add_argument("--n_neighbors", required=False, default=5,)
    knc.add_argument("--weights",required=False, default='uniform') 
    knc.add_argument("--algorithm", required=False, default='auto' )
    knc.add_argument("--leaf_size", required=False, default=30, )
    knc.add_argument("--p", required=False, default=2, )
    knc.add_argument("--metric", required=False, default='minkowski', )
    knc.add_argument("--metric_params", required=False, default=None )
    knc.add_argument("--n_jobs", required=False, default=None,)
    knc.add_argument("--TrainFile", required=True, default=None, help="")
    knc.add_argument("--TestMethod", required=True, default=None, help="")
    knc.add_argument("--SelectedSclaer", required=True, help="")
    knc.add_argument("--NFolds", required=False, default=5, help="")
    knc.add_argument("--Testspt", required=False, default=0.2, help="")
    knc.add_argument("--TestFile", required=False, default=None, help="")
    knc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    knc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    knc.add_argument("--htmlFname", required=False, help="")
    knc.add_argument("--OfileName", required=False, help="")
    knc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    gnbc = subparsers.add_parser('GNBC')
    gnbc.add_argument("--priors", required=False, default=None)
    gnbc.add_argument("--var_smoothing", required=False, default=1e-09)
    gnbc.add_argument("--TrainFile", required=True, default=None, help="")
    gnbc.add_argument("--TestMethod", required=True, default=None, help="")
    gnbc.add_argument("--SelectedSclaer", required=True, help="")
    gnbc.add_argument("--NFolds", required=False, default=5, help="")
    gnbc.add_argument("--Testspt", required=False, default=0.2, help="")
    gnbc.add_argument("--TestFile", required=False, default=None, help="")
    gnbc.add_argument("--OutFile", required=False, default='Out.csv', help="")
    gnbc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="MinMaxScaler")
    gnbc.add_argument("--htmlFname", required=False, help="")
    gnbc.add_argument("--OfileName", required=False, help="")
    gnbc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="")

    args = parser.parse_args()

if   sys.argv[1] == 'SVMC':
    SVM_Classifier(args.C, args.kernel, args.degree, args.gamma, args.coef0, args.shrinking, args.probability, args.tol, args.cache_size, args.class_weight, args.verbose, args.max_iter, args.decision_function_shape, args.randomRtate, args.TrainFile, args.TestMethod,  args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'SGDC':   
    SGD_Classifier( args.loss, args.penalty, args.alpha, args.l1_ratio, args.fit_intercept, args.max_iter, args.tol, args.shuffle, args.verbose, args.epsilon, args.n_jobs, args.random_state, args.learning_rate, args.eta0, args.power_t, args.early_stopping, args.validation_fraction, args.n_iter_no_change, args.class_weight, args.warm_start, args.average, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'DTC':
    DT_Classifier(args.criterion, args.splitter, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.min_weight_fraction_leaf, args.max_features,  args.random_state, args.max_leaf_nodes, args.min_impurity_decrease, args.min_impurity_split, args.class_weight, args.presort, args.ccpalpha, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'GBC':
    GB_Classifier(args.loss, args.learning_rate, args.n_estimators, args.subsample, args.criterion, args.min_samples_split, args.min_samples_leaf, args.min_weight_fraction_leaf,  args.max_depth, args.min_impurity_decrease, args.min_impurity_split, args.init, args.random_state, args.max_features, args.verbose, args.max_leaf_nodes, args.warm_start, args.presort, args.validation_fraction, args.n_iter_no_change, args.tol, args.ccpalpha, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'RFC':
    RF_Classifier( args.n_estimators, args.criterion, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.min_weight_fraction_leaf, args.max_features, args.max_leaf_nodes, args.min_impurity_decrease,  args.min_impurity_split, args.bootstrap, args.oob_score, args.n_jobs, args.random_state, args.verbose, args.warm_start, args.class_weight, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'LRC':
    LR_Classifier(args.penalty, args.dual, args.tol, args.C, args.fit_intercept, args.intercept_scaling, args.class_weight, args.random_state, args.solver, args.max_iter, args.multi_class, args.verbose, args.warm_start, args.n_jobs, args.l1_ratio, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'KNC':
    KN_Classifier(args.n_neighbors, args.weights, args.algorithm, args.leaf_size, args.p, args.metric, args.metric_params,  args.n_jobs, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'GNBC':
    GNB_Classifier(args.priors, args.var_smoothing, args.TrainFile, args.TestMethod,  args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)  
else:
    print ("its not accurate")
    exit()




