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
################################################################

def HTML_Gen(html):

    out_html = open(html,'w')             
    part_1 =  """

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Bootstrap Example</title>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    <body>
    <style>
    div.container_1 {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    div.table {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    </style>
    </head>
    <div class="jumbotron text-center">
      <h1> Machine Learning Algorithm Assessment Report </h1>
    </div>
    <div class="container">
      <h2> ROC curve and result summary Graph </h2>
      <div class="row">
        <div class="col-sm-4">
          <img src="2.jpg" alt="Smiley face" height="350" width="350">
        </div>
        <div class="col-sm-4">
          <img src="out.jpg" alt="Smiley face" height="350" width="350">
        </div>
      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()

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

def Fit_Model(TrainData, Test_Method, Algo, Selected_Sclaer,  Workdirpath,  htmlOutDir, OutFile, htmlFname,    NoOfFolds=None, TestSize=None, TestData=None ):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    if Test_Method == 'Internal':
        X,y,_,_ = ReturnData(TrainData, Test_Method)

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

        ##########################
        accuracy_score_l = []
        cohen_kappa_score_l = []
        matthews_corrcoef_l = []
        precision_l = []
        recall_l = []
        f_score_l = []
        ##########################

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

        mean_tpr /= folds.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        pl.plot(mean_fpr, mean_tpr, '-', color='red',label='AUC = %0.2f' % mean_auc, lw=2)

        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC Cureve for All the classifier')
        pl.legend(loc="lower right")

        ########################################################################################################################################
        V_header = ["accuracy","presision","recall","f1","mean_auc"]                                                                           #
        v_values = [round(accuracy_score_mean, 3), round(precision_mean, 3), round(recall_mean, 3),round(f_score_mean, 3), round(mean_auc, 3)] #                                         # 
        ########################################################################################################################################

        df = pd.DataFrame([v_values], columns=V_header)
        df.to_csv(os.path.join(Workdirpath, OutFile), columns=V_header)
        pl.savefig(os.path.join(Workdirpath, htmlOutDir, "out.jpg"))
        pl.figure()
        pl.bar(V_header, v_values, color=(0.2, 0.4, 0.6, 0.6))
        pl.xlabel('Accuracy Perameters', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')
        pl.savefig(os.path.join(Workdirpath, htmlOutDir, "2.jpg"))
        #pl.show()
        HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))
        print "Internal"

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

        prob = Algo.fit(x_train, y_train).predict_proba(x_test)
        predicted = Algo.fit(x_train, y_train).predict(x_test)

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

        pl.figure()
        pl.plot(fpr, tpr, '-', color='red',label='AUC = %0.2f' % auc_score, lw=2)
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC Cureve for All the classifier')
        pl.legend(loc="lower right")

        df = pd.DataFrame([v_values], columns=V_header)
        pl.savefig(os.path.join(Workdirpath, htmlOutDir, "out.jpg"))

        pl.figure()
        pl.bar(V_header, v_values, color=(0.2, 0.4, 0.6, 0.6))
        pl.xlabel('Accuracy Perameters', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')
        pl.savefig(os.path.join(Workdirpath, htmlOutDir, "2.jpg"))
        #pl.show()
        HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))

        print "External"

    elif Test_Method == "TestSplit":

        X_train,y_train,_,_ = ReturnData(TrainData, Test_Method)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TestSize, random_state=0)

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

        pl.figure()
        pl.bar(V_header, v_values, color=(0.2, 0.4, 0.6, 0.6))
        pl.xlabel('Accuracy Perameters', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')
        pl.savefig(os.path.join(Workdirpath, htmlOutDir, "2.jpg"))
        #pl.show()
        HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))
        print "TestSplit"

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

        print "TestSplit"

        return predicted

def SVM_Classifier(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, randomState, TrainFile, TestMethod,  SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):
   
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
    'random_state':randomState
    }

    model = SVC(**pera )

    print Fit_Model
    
    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname,  NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def SGD_Classifier( loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon, n_jobs, random_state, learning_rate, eta0, power_t, early_stopping, validation_fraction, n_iter_no_change, class_weight, warm_start, average, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):

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

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname,  NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def DT_Classifier(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,  random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, presort, ccpalpha, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):

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

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname,  NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def GB_Classifier(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,  max_depth, min_impurity_decrease,min_impurity_split, init, random_state, max_features, verbose, max_leaf_nodes, warm_start, presort, validation_fraction, n_iter_no_change, tol, ccpalpha, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):

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

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname,  NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def RF_Classifier( n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease,  min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, class_weight, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):

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

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname,  NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def LR_Classifier(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio, TrainFile, TestMethod, SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):

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

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname,  NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def KN_Classifier(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params,  n_jobs, TrainFile, TestMethod,  SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):

    pera = {"weights":weights, 
    "algorithm":algorithm, 
    "leaf_size":leaf_size, 
    "p":p, "metric":metric, 
    "metric_params":metric_params, 
    "n_jobs":n_jobs}
    
    model = KNeighborsClassifier(**pera)

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname,  NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)

def GNB_Classifier( priors, var_smoothing, TrainFile, TestMethod,  SelectedSclaer, NFolds, Testspt, TestFile, OutFile, htmlOutDir, htmlFname, Workdirpath):  

    pera = {"priors":priors, 
    "var_smoothing":var_smoothing}  

    model = GaussianNB(**pera) 

    Fit_Model(TrainData=TrainFile, Test_Method=TestMethod, Algo=model, Selected_Sclaer=SelectedSclaer, Workdirpath=Workdirpath, htmlOutDir=htmlOutDir, OutFile=OutFile, htmlFname=htmlFname, NoOfFolds=NFolds, TestSize=Testspt, TestData=TestFile)
  
if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    svmc = subparsers.add_parser('SVMC')
    svmc.add_argument("--C", required=False, default=1.0, help="Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.")
    svmc.add_argument("--kernel", required=False, default='rbf', help="Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).")
    svmc.add_argument("--degree", required=False, default=3, help="Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.")
    svmc.add_argument("--gamma", required=False, default='auto_deprecated', help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, if 'auto', uses 1 / n_features.")
    svmc.add_argument("--coef0", required=False, default=0.0, help="Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.")
    svmc.add_argument("--shrinking", required=False, default=True, help="Whether to use the shrinking heuristic.")
    svmc.add_argument("--probability", required=False, default=False, help="Whether to enable probability estimates. This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with predict")
    svmc.add_argument("--tol", required=False, default=0.001, help="Tolerance for stopping criterion.")
    svmc.add_argument("--cache_size", required=False, default=200, help="Specify the size of the kernel cache (in MB).")
    svmc.add_argument("--class_weight", required=False, default=None, help="Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))")
    svmc.add_argument("--verbose", required=False, default=False, help="Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.")
    svmc.add_argument("--max_iter", required=False, default=-1, help="Hard limit on iterations within solver, or -1 for no limit.")
    svmc.add_argument("--decision_function_shape", required=False, default='ovr', help="Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one ('ovo') is always used as multi-class strategy.")
    svmc.add_argument("--randomState", required=False, default=None, help="The seed of the pseudo random number generator used when shuffling the data for probability estimates. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.") 
    svmc.add_argument("--breakties", required=False, default=False, help="If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple predict." )
    svmc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    svmc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    svmc.add_argument("--SelectedSclaer", required=True, help="'Min_Max','Standard_Scaler','No_Scaler'")
    svmc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    svmc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    svmc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    svmc.add_argument("--OutFile", required=False, default='Out.csv', help="Out.csv")
    svmc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    svmc.add_argument("--htmlFname", required=False, default='Out.html', help="HTML out file")
    svmc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    sgdc = subparsers.add_parser('SGDC')
    sgdc.add_argument("--loss", required=False, default='log', help="The loss function to be used. Defaults to 'hinge', which gives a linear SVM. The possible options are 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', or a regression loss: 'squared_loss', 'huber', 'epsilon_insensitive', or squared_epsilon_insensitive'.")
    sgdc.add_argument("--penalty", required=False, default='l2', help="The penalty (aka regularization term) to be used. Defaults to 'l2' which is the standard regularizer for linear SVM models. 'l1' and 'elasticnet' might bring sparsity to the model (feature selection) not achievable with 'l2'.")
    sgdc.add_argument("--alpha", required=False, default=0.0001, help="Constant that multiplies the regularization term. Defaults to 0.0001. Also used to compute learning_rate when set to 'optimal'.")
    sgdc.add_argument("--l1_ratio", required=False, default=0.15, help="The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.")
    sgdc.add_argument("--fit_intercept", required=False, default=True, help="Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.")
    sgdc.add_argument("--max_iter", required=False, default=1000, help="The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.")
    sgdc.add_argument("--tol", required=False, default=0.001, help="The stopping criterion. If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.")
    sgdc.add_argument("--shuffle", required=False, default=True, help="Whether or not the training data should be shuffled after each epoch. Defaults to True.")
    sgdc.add_argument("--verbose", required=False, default=0, help="The verbosity level.")
    sgdc.add_argument("--epsilon", required=False, default=0.1, help="Epsilon in the epsilon-insensitive loss functions; only if loss is 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'. For 'huber', determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.")
    sgdc.add_argument("--n_jobs", required=False, default=None, help="The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.")
    sgdc.add_argument("--random_state", required=False, default=None, help="The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")
    sgdc.add_argument("--learning_rate", required=False, default='optimal', help="The learning rate schedule:")
    sgdc.add_argument("--eta0", required=False, default=0.0, help="eta = eta0")
    sgdc.add_argument("--power_t", required=False, default=0.5, help="eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.")
    sgdc.add_argument("--early_stopping", required=False, default=False, help="MinMaxScaler")
    sgdc.add_argument("--validation_fraction", required=False, default=0.1, help="The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.")
    sgdc.add_argument("--n_iter_no_change", required=False, default=5, help="Number of iterations with no improvement to wait before early stopping.")
    sgdc.add_argument("--class_weight", required=False, default=None, help="Preset for the class_weight fit parameter. Weights associated with classes. If not given, all classes are supposed to have weight one. The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).")
    sgdc.add_argument("--warm_start", required=False, default=False, help="When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.")
    sgdc.add_argument("--average", required=False, default=False, help="MinMaxScaler")
    sgdc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    sgdc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    sgdc.add_argument("--SelectedSclaer", required=True, help="'Min_Max','Standard_Scaler','No_Scaler'")
    sgdc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    sgdc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    sgdc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    sgdc.add_argument("--OutFile", required=False, default='Out.csv', help="float, Max=1.0")
    sgdc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    sgdc.add_argument("--htmlFname", required=False, help="")
    sgdc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    dtc = subparsers.add_parser('DTC') 
    dtc.add_argument("--criterion", required=False, default='gini', help="The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain.")
    dtc.add_argument("--splitter", required=False, default='best', help="The strategy used to choose the split at each node. Supported strategies are 'best' to choose the best split and 'random' to choose the best random split." )
    dtc.add_argument("--max_depth", required=False, default=None, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    dtc.add_argument("--min_samples_split", required=False, default=2, help="The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.")
    dtc.add_argument("--min_samples_leaf", required=False, default=1, help="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.")
    dtc.add_argument("--min_weight_fraction_leaf", required=False, default=0.0, help="The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.")
    dtc.add_argument("--max_features", required=False, default=None, help="The number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features.")
    dtc.add_argument("--random_state", required=False, default=None, help="If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")
    dtc.add_argument("--max_leaf_nodes", required=False, default=None, help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value. The weighted impurity decrease equation is the following")
    dtc.add_argument("--min_impurity_decrease", required=False, default=0.0, help="")
    dtc.add_argument("--min_impurity_split", required=False, default=None, help="Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.")
    dtc.add_argument("--class_weight", required=False, default=None, help="Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y. Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}]. The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)) For multi-output, the weights of each column of y will be multiplied. Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.")
    dtc.add_argument("--presort", required=False, default=False, help="This parameter is deprecated and will be removed in v0.24.")
    dtc.add_argument("--ccpalpha", required=False, default=0.0, help="Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.")
    dtc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    dtc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    dtc.add_argument("--SelectedSclaer", required=True, help="'Min_Max',Standard_Scaler','No_Scaler'")
    dtc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    dtc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    dtc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    dtc.add_argument("--OutFile", required=False, default='Out.csv', help="Out.tsv")
    dtc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    dtc.add_argument("--htmlFname", required=False, help="HTML out file")
    dtc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    gbc =  subparsers.add_parser('GBC')
    gbc.add_argument("--loss", required=False, default='deviance', help="loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm.") 
    gbc.add_argument("--learning_rate", required=False, default=0.1, help="learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.") 
    gbc.add_argument("--n_estimators", required=False, default=100, help="The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.") 
    gbc.add_argument("--subsample", required=False, default=1.0, help="The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.") 
    gbc.add_argument("--criterion", required=False,default='friedman_mse', help="The function to measure the quality of a split. Supported criteria are 'friedman_mse' for the mean squared error with improvement score by Friedman, 'mse' for mean squared error, and 'mae' for the mean absolute error. The default value of 'friedman_mse' is generally the best as it can provide a better approximation in some cases.") 
    gbc.add_argument("--min_samples_split", required=False, default=2, help="The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.") 
    gbc.add_argument("--min_samples_leaf", required=False, default=1, help="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.") 
    gbc.add_argument("--min_weight_fraction_leaf", required=False, default=0.0, help="The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.") 
    gbc.add_argument("--max_depth", required=False, default=3, help="maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.") 
    gbc.add_argument("--min_impurity_decrease", required=False, default=0.0, help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value. The weighted impurity decrease equation is the following: 'N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity'), where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child. N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed. New in version 0.19.")
    gbc.add_argument("--min_impurity_split", required=False, default=None, help="Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.") 
    gbc.add_argument("--init", required=False,default=None, help="An estimator object that is used to compute the initial predictions. init has to provide fit and predict_proba. If 'zero', the initial raw predictions are set to zero. By default, a DummyEstimator predicting the classes priors is used.") 
    gbc.add_argument("--random_state", required=False, default=None, help="If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.") 
    gbc.add_argument("--max_features", required=False, default=None, help="The number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features. Choosing max_features < n_features leads to a reduction of variance and an increase in bias. Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.") 
    gbc.add_argument("--verbose",required=False, default=0, help="Enable verbose output. If 1 then it prints progress and performance once in a while (the more trees the lower the frequency). If greater than 1 then it prints progress and performance for every tree.") 
    gbc.add_argument("--max_leaf_nodes", required=False, default=None, help="Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.") 
    gbc.add_argument("--warm_start", required=False, default=False, help="When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution." ) 
    gbc.add_argument("--presort", required=False,default='auto', help="This parameter is deprecated and will be removed in v0.24.") 
    gbc.add_argument("--validation_fraction", required=False, default=0.1, help="The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.") 
    gbc.add_argument("--n_iter_no_change", required=False, default=None, help="n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations. The split is stratified.") 
    gbc.add_argument("--tol", required=False, default=0.0001, help="Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.")
    gbc.add_argument("--ccpalpha", required=False, default=0.0, help="Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.")
    gbc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    gbc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    gbc.add_argument("--SelectedSclaer", required=True, help="'Min_Max',Standard_Scaler','No_Scaler'")
    gbc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    gbc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    gbc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    gbc.add_argument("--OutFile", required=False, default='Out.csv', help="Out.tsv")
    gbc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    gbc.add_argument("--htmlFname", required=False, help="HTML out file")
    gbc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    rfc = subparsers.add_parser('RFC')
    rfc.add_argument("--n_estimators", required=False, default=100, help="The number of trees in the forest.") 
    rfc.add_argument("--criterion", required=False, default='gini', help="The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain. Note: this parameter is tree-specific." ) 
    rfc.add_argument("--max_depth", required=False, default=None, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.") 
    rfc.add_argument("--min_samples_split", required=False, default=2, help="The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.") 
    rfc.add_argument("--min_samples_leaf", required=False, default=1, help="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.") 
    rfc.add_argument("--min_weight_fraction_leaf", required=False, default=0.0, help="The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.")
    rfc.add_argument("--max_features", required=False, default='auto', help="The number of features to consider when looking for the best split:") 
    rfc.add_argument("--max_leaf_nodes", required=False, default=None, help="Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.")
    rfc.add_argument("--min_impurity_decrease", required=False, default=0.0, help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value. The weighted impurity decrease equation is the following: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity) where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child. N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed. New in version 0.19.")  
    rfc.add_argument("--min_impurity_split", required=False, default=None, help="Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.") 
    rfc.add_argument("--bootstrap", required=False, default=True, help="Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.")
    rfc.add_argument("--oob_score", required=False, default=False, help="Whether to use out-of-bag samples to estimate the generalization accuracy.")
    rfc.add_argument("--n_jobs", required=False, default=None, help="The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details." )
    rfc.add_argument("--random_state", required=False, default=None, help="Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). See Glossary for details.") 
    rfc.add_argument("--verbose", required=False, default=0, help="Controls the verbosity when fitting and predicting." )
    rfc.add_argument("--warm_start", required=False, default=False, help="When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See the Glossary.") 
    rfc.add_argument("--class_weight", required=False, default=None, help="Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y. Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of ['{1:1}, {2:5}, {3:1}, {4:1}']. The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)) The 'balanced_subsample' mode is the same as 'balanced' except that weights are computed based on the bootstrap sample for every tree grown. For multi-output, the weights of each column of y will be multiplied. Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.")
    rfc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    rfc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    rfc.add_argument("--SelectedSclaer", required=True, help="'Min_Max',Standard_Scaler','No_Scaler'")
    rfc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    rfc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    rfc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    rfc.add_argument("--OutFile", required=False, default='Out.csv', help="Out.tsv")
    rfc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    rfc.add_argument("--htmlFname", required=False, help="HTML out file")
    rfc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    lrc =  subparsers.add_parser('LRC')
    lrc.add_argument("--penalty", required=False, default='l2', help="Used to specify the norm used in the penalization. The 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is only supported by the 'saga' solver. If 'none' (not supported by the liblinear solver), no regularization is applied." )
    lrc.add_argument("--dual", required=False, default=False, help="Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.")
    lrc.add_argument("--tol", required=False, default=0.0001, help="Tolerance for stopping criteria.")
    lrc.add_argument("--C", required=False, default=1.0, help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization." )
    lrc.add_argument("--fit_intercept", required=False, default=True, help="Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function." )
    lrc.add_argument("--intercept_scaling", required=False, default=1, help="Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a 'synthetic' feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight." )
    lrc.add_argument("--class_weight", required=False, default=None, help="Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified. New in version 0.17: class_weight='balanced'")
    lrc.add_argument("--random_state", default=None, help="The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when solver == 'sag' or 'liblinear'.") 
    lrc.add_argument("--solver", required=False, default='lbfgs', help="Algorithm to use in the optimization problem. For small datasets, 'liblinear' is a good choice, whereas 'sag' and 'saga' are faster for large ones. For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss; 'liblinear' is limited to one-versus-rest schemes. 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty 'liblinear' and 'saga' also handle L1 penalty 'saga' also supports 'elasticnet' penalty 'liblinear' does not support setting penalty='none' Note that 'sag' and 'saga' fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing. New in version 0.17: Stochastic Average Gradient descent solver.") 
    lrc.add_argument("--max_iter", required=False, default=100, help="Maximum number of iterations taken for the solvers to converge."), 
    lrc.add_argument("--multi_class", required=False, default='auto', help="If the option chosen is 'ovr', then a binary problem is fit for each label. For 'multinomial' the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. 'multinomial' is unavailable when solver='liblinear'. 'auto' selects 'ovr' if the data is binary, or if solver='liblinear', and otherwise selects 'multinomial'. New in version 0.18: Stochastic Average Gradient descent solver for 'multinomial' case.") 
    lrc.add_argument("--verbose", required=False, default=0, help="For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.")
    lrc.add_argument("--warm_start", required=False, default=False, help="When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver. See the Glossary. New in version 0.17: warm_start to support lbfgs, newton-cg, sag, saga solvers.")
    lrc.add_argument("--n_jobs", required=False, default=None, help="Number of CPU cores used when parallelizing over classes if multi_class='ovr'. This parameter is ignored when the solver is set to 'liblinear' regardless of whether 'multi_class' is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details." )
    lrc.add_argument("--l1_ratio", required=False, default=None, help="The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting 'l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.")
    lrc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    lrc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    lrc.add_argument("--SelectedSclaer", required=True, help="'Min_Max',Standard_Scaler','No_Scaler'")
    lrc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    lrc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    lrc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    lrc.add_argument("--OutFile", required=False, default='Out.csv', help="Out.tsv")
    lrc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    lrc.add_argument("--htmlFname", required=False,  help="HTML out file")
    lrc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    knc = subparsers.add_parser('KNC')
    knc.add_argument("--n_neighbors", required=False, default=5, help="Number of neighbors to use by default for kneighbors queries.")
    knc.add_argument("--weights",required=False, default='uniform', help="weight function used in prediction. Possible values: 'uniform' : uniform weights. All points in each neighborhood are weighted equally. 'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away. [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.") 
    knc.add_argument("--algorithm", required=False, default='auto', help="Algorithm used to compute the nearest neighbors:'ball_tree' will use BallTree 'kd_tree' will use KDTree 'brute' will use a brute-force search. 'auto' will attempt to decide the most appropriate algorithm based on the values passed to fit method. Note: fitting on sparse input will override the setting of this parameter, using brute force." )
    knc.add_argument("--leaf_size", required=False, default=30, help="Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.")
    knc.add_argument("--p", required=False, default=2, help="Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used." )
    knc.add_argument("--metric", required=False, default='minkowski', help="the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric class for a list of available metrics. If metric is 'precomputed', X is assumed to be a distance matrix and must be square during fit. X may be a Glossary, in which case only 'nonzero' elements may be considered neighbors.")
    knc.add_argument("--metric_params", required=False, default=None, help="Additional keyword arguments for the metric function." )
    knc.add_argument("--n_jobs", required=False, default=None, help="The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details. Doesn't affect fit method.")
    knc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    knc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    knc.add_argument("--SelectedSclaer", required=True, help="'Min_Max',Standard_Scaler','No_Scaler'")
    knc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    knc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    knc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    knc.add_argument("--OutFile", required=False, default='Out.csv', help="Out.tsv")
    knc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    knc.add_argument("--htmlFname", required=False, help="HTML out file")
    knc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    gnbc = subparsers.add_parser('GNBC')
    gnbc.add_argument("--priors", required=False, default=None, help="Prior probabilities of the classes. If specified the priors are not adjusted according to the data.")
    gnbc.add_argument("--var_smoothing", required=False, default=1e-09, help="Portion of the largest variance of all features that is added to variances for calculation stability.")
    gnbc.add_argument("--TrainFile", required=True, default=None, help="Positive negative dataset Ex. 'Train.csv'")
    gnbc.add_argument("--TestMethod", required=True, default=None, help="Internal','CrossVal', 'External', 'Predict'")
    gnbc.add_argument("--SelectedSclaer", required=True, help="'Min_Max',Standard_Scaler','No_Scaler'")
    gnbc.add_argument("--NFolds", required=False, default=5, help="int, Max=10")
    gnbc.add_argument("--Testspt", required=False, default=0.2, help="float, Max=1.0")
    gnbc.add_argument("--TestFile", required=False, default=None, help="Test data, 'Test.csv'")
    gnbc.add_argument("--OutFile", required=False, default='Out.csv', help="Out.tsv")
    gnbc.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    gnbc.add_argument("--htmlFname", required=False, help="HTML out file")
    gnbc.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

    args = parser.parse_args()

if   sys.argv[1] == 'SVMC':
    SVM_Classifier(args.C, args.kernel, args.degree, args.gamma, args.coef0, args.shrinking, args.probability, args.tol, args.cache_size, args.class_weight, args.verbose, args.max_iter, args.decision_function_shape, args.randomState, args.TrainFile, args.TestMethod,  args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.Workdirpath)
elif sys.argv[1] == 'SGDC':   
    SGD_Classifier( args.loss, args.penalty, args.alpha, args.l1_ratio, args.fit_intercept, args.max_iter, args.tol, args.shuffle, args.verbose, args.epsilon, args.n_jobs, args.random_state, args.learning_rate, args.eta0, args.power_t, args.early_stopping, args.validation_fraction, args.n_iter_no_change, args.class_weight, args.warm_start, args.average, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.Workdirpath)
elif sys.argv[1] == 'DTC':
    DT_Classifier(args.criterion, args.splitter, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.min_weight_fraction_leaf, args.max_features,  args.random_state, args.max_leaf_nodes, args.min_impurity_decrease, args.min_impurity_split, args.class_weight, args.presort, args.ccpalpha, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.OfileName, args.Workdirpath)
elif sys.argv[1] == 'GBC':
    GB_Classifier(args.loss, args.learning_rate, args.n_estimators, args.subsample, args.criterion, args.min_samples_split, args.min_samples_leaf, args.min_weight_fraction_leaf,  args.max_depth, args.min_impurity_decrease, args.min_impurity_split, args.init, args.random_state, args.max_features, args.verbose, args.max_leaf_nodes, args.warm_start, args.presort, args.validation_fraction, args.n_iter_no_change, args.tol, args.ccpalpha, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.Workdirpath)
elif sys.argv[1] == 'RFC':
    RF_Classifier( args.n_estimators, args.criterion, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.min_weight_fraction_leaf, args.max_features, args.max_leaf_nodes, args.min_impurity_decrease,  args.min_impurity_split, args.bootstrap, args.oob_score, args.n_jobs, args.random_state, args.verbose, args.warm_start, args.class_weight, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.Workdirpath)
elif sys.argv[1] == 'LRC':
    LR_Classifier(args.penalty, args.dual, args.tol, args.C, args.fit_intercept, args.intercept_scaling, args.class_weight, args.random_state, args.solver, args.max_iter, args.multi_class, args.verbose, args.warm_start, args.n_jobs, args.l1_ratio, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.Workdirpath)
elif sys.argv[1] == 'KNC':
    KN_Classifier(args.n_neighbors, args.weights, args.algorithm, args.leaf_size, args.p, args.metric, args.metric_params,  args.n_jobs, args.TrainFile, args.TestMethod, args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.Workdirpath)
elif sys.argv[1] == 'GNBC':
    GNB_Classifier(args.priors, args.var_smoothing, args.TrainFile, args.TestMethod,  args.SelectedSclaer, args.NFolds, args.Testspt, args.TestFile, args.OutFile, args.htmlOutDir, args.htmlFname, args.Workdirpath)  
else:
    print ("its not accurate")
    exit()