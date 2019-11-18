
def SVM_classification(InData,

    C,
    kernel
    degree,
    gamma,
    coef0,
    shrinking,
    probability,
    tol,
    cache_size,
    class_weight,
    verbose,
    max_iter,
    decision_function_shape,
    random_state,
    OutDir):

def SGD_Classification(InData, 
    loss 
    penalty 
    alpha
    l1_ratio
    fit_intercept, 
    max_iter, 
    tol, 
    shuffle, 
    verbose, 
    epsilon, 
    n_jobs, 
    random_state, 
    learning_rate, 
    eta0, 
    power_t, 
    early_stopping, 
    validation_fraction, 
    n_iter_no_change, 
    class_weight, 
    warm_start, 
    average,
    OutDir):

def DecisionTree_Classification(InData,
    criterion,
    splitter, 
    max_depth, 
    min_samples_split, 
    min_samples_leaf, 
    min_weight_fraction_leaf, 
    max_features, 
    random_state, 
    max_leaf_nodes, 
    min_impurity_decrease, 
    min_impurity_split, 
    class_weight, 
    presort,
    OutDir):

def GradientBoosting_Classification(InData,
    loss, 
    learning_rate, 
    n_estimators, 
    subsample, 
    criterion, 
    min_samples_split, 
    min_samples_leaf, 
    min_weight_fraction_leaf, 
    max_depth, 
    min_impurity_decrease,
    min_impurity_split, 
    init, 
    random_state, 
    max_features, 
    verbose, 
    max_leaf_nodes, 
    warm_start, 
    presort, 
    validation_fraction, 
    n_iter_no_change, 
    tol,
    OutDir):

def RandomForestClassifier(InData, 
    n_estimators, 
    criterion, 
    max_depth, 
    min_samples_split, 
    min_samples_leaf, 
    min_weight_fraction_leaf, 
    max_features, 
    max_leaf_nodes, 
    min_impurity_decrease, 
    min_impurity_split, 
    bootstrap, 
    oob_score, 
    n_jobs, 
    random_state, 
    verbose, 
    warm_start, 
    class_weight,
    OutDir):

def LogisticRegression(InData,
    penalty, 
    dual, 
    tol, 
    C, 
    fit_intercept, 
    intercept_scaling, 
    class_weight, 
    random_state, 
    solver, 
    max_iter, 
    multi_class, 
    verbose, 
    warm_start 
    n_jobs, 
    l1_ratio,
    OutDir):

def KNeighbors_Classifier(n_neighbors,
    weights, 
    algorithm, 
    leaf_size, 
    p, 
    metric, 
    metric_params, 
    n_jobs,
    OutDir):

def GaussianNB(InData, 
    priors, 
    var_smoothing, 
    OutDir):    

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
