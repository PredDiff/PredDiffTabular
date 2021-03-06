import numpy as np
import pandas as pd
import scipy
import scipy.stats as st

from typing import List, Union

from tqdm.auto import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer as IterativeImputerOriginal

from .IterativeImputerPR import IterativeImputer as IterativeImputerPR
from ..tools.utils_bootstrap import empirical_bootstrap
# from ..tools.bootstrap_utils import empirical_bootstrap

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor


##########################################################################################
#EVALUATION
##########################################################################################
def _eval_mean(predictions, weights=None, log_transform=False, n_train_samples=10000, n_classes = 10):
    if(weights is None):
        mean = predictions.mean(axis=1)
    else:
        mean = np.sum(predictions*weights, axis=1)/np.sum(weights, axis=1)
    return mean


def evaluate_imputer(df,x,imputer_name=None,n_bootstrap=100,verbose=False):
    '''evaluates imputers based on metrics from https://stefvanbuuren.name/fimd/sec-evaluation.html and some basic statistical tests
    Parameters:
    df: original dataframe
    x: result of applying imputer to df
    '''
    res = []
    for xi in x:
        cols = [a for a in xi.columns if a not in ['id','imputation_id','sampling_prob']]
        for c in cols:
            imputations = np.stack(list(xi.groupby('id')[c].apply(lambda y: np.array(y))),axis=0)#n_samples,n_imputations
            if("sampling_prob" in xi.columns):
                sampling_probs =np.stack(list(xi.groupby('id')["sampling_prob"].apply(lambda y: np.array(y))),axis=0)
            else:
                sampling_probs = None
            ground_truth = np.array(df[c])
            
            mean_pred, bounds_low, bounds_high, _ = empirical_bootstrap(imputations if sampling_probs is None else (imputations,sampling_probs), _eval_mean, n_iterations=n_bootstrap)
            
            raw_bias = np.abs(mean_pred- ground_truth)
            percentage_bias = np.mean(np.abs(raw_bias/(ground_truth+1e-8)))
            raw_bias = np.mean(raw_bias)
            if(verbose):
                print("\n\nimputed cols:",cols," var:",c,"dtype:",df[c].dtype)
                print("for reference: mean",np.mean(ground_truth),"std",np.std(ground_truth))
                print("mean raw bias:",np.mean(raw_bias))
                print("mean percentage bias:",np.mean(percentage_bias))
            
            #print(bounds_low[:3],bounds_high[:3],mean_pred[:3],ground_truth[:3])
            coverage_rate = np.mean(np.logical_and(bounds_low <= ground_truth, ground_truth <= bounds_high))
            
            average_width = np.mean(bounds_high-bounds_low)
            
            rmse = np.sqrt(np.mean(np.power(ground_truth-mean_pred,2)))
            try:
                mwu =  st.mannwhitneyu(ground_truth,mean_pred).pvalue #low p value: reject H0 that both are sampled from the same distribution
            except:
                mwu = np.nan #all identical
            try:
                wc= st.wilcoxon(ground_truth,mean_pred).pvalue
            except:
                wc = np.nan #Wilcoxon pvalue could not be calculated due to constant predictions
            
            if(verbose):
                print("coverage rate:",coverage_rate)
                print("average width:",average_width)
                print("rmse:",rmse)
                print("Mann Whitney U pvalue:", mwu)
                print("Wilcoxon pvalue:",wc)
            
            res.append({"imputed_cols":cols,"var":c,"dtype":df[c].dtype,"gt_mean":np.mean(ground_truth),"gt_std":np.std(ground_truth),"pred_mean":np.mean(mean_pred),"pred_std":np.std(mean_pred),"raw_bias":raw_bias,"percentage_bias":percentage_bias,"coverage_rate":coverage_rate,"average_width":average_width,"rmse":rmse,"mann-whitney_u_p":mwu,"wilcoxon_p":wc})
        
    df=pd.DataFrame(res)
    if(imputer_name is not None):
        df["imputer"]=imputer_name
    return df


##########################################################################################
# RFxxxSampling/ETxxxSampling
##########################################################################################
class RandomForestRegressorSampling(RandomForestRegressor):
    '''RandomForestRegressor that samples from a Gaussian distribution based on mean/std of trees'''
    def predict(self, X):
        #possibility to speed this up by mimicing the original predict #https://stackoverflow.com/questions/20615750/how-do-i-output-the-regression-prediction-from-each-tree-in-a-random-forest-in-p?rq=1
        per_tree_pred = [tree.predict(X) for tree in self.estimators_]
        sample_mean = np.mean(per_tree_pred,axis=0)
        sample_std = np.std(per_tree_pred,axis=0)
        return sample_mean+sample_std* np.random.randn(len(X))


class RandomForestClassifierSampling(RandomForestClassifier):
    '''RandomForestClassifier that samples from the predict_proba multinomial distribution'''
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array([np.where(np.random.multinomial(1, ps))[0] for ps in probs]).astype(np.int64)


class ExtraTreesRegressorSampling(ExtraTreesRegressor):
    '''ExtraTreesRegressor that samples from a Gaussian distribution based on mean/std of trees'''
    def predict(self, X):
        #possibility to speed this up by mimicing the original predict #https://stackoverflow.com/questions/20615750/how-do-i-output-the-regression-prediction-from-each-tree-in-a-random-forest-in-p?rq=1
        per_tree_pred = [tree.predict(X) for tree in self.estimators_]
        sample_mean = np.mean(per_tree_pred,axis=0)
        sample_std = np.std(per_tree_pred,axis=0)
        return sample_mean+sample_std* np.random.randn(len(X))


class ExtraTreesClassifierSampling(ExtraTreesClassifier):
    '''RandomForestClassifier that samples from the predict_proba multinomial distribution'''
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array([np.where(np.random.multinomial(1, ps))[0] for ps in probs]).astype(np.int64)


#########################################################################################
# Imputer Baseclass
#########################################################################################
class ImputerBase(object):
    '''Imputer Base class; takes care of label encoding, standard scaling'''
    def __init__(self, df_train, **kwargs):
        self.df_train = df_train.copy()
        self.kwargs = kwargs

        self.exclude_cols = self.kwargs["exclude_cols"] if "exclude_cols" in self.kwargs.keys() else []
        self.nocat_cols = self.kwargs["nocat_cols"] if "nocat_cols" in self.kwargs.keys() else []
        #label-encode by default
        label_encode = self.kwargs["label_encode"] if "label_encode" in self.kwargs.keys() else True
        standard_scale = self.kwargs["standard_scale"] if "standard_scale" in self.kwargs.keys() else False
        standard_scale_all = self.kwargs["standard_scale_all"] if "standard_scale_all" in self.kwargs.keys() else False
        self.categorical_columns = [x for x in self.df_train.columns[np.where(np.logical_and(self.df_train.dtypes != np.float64,self.df_train.dtypes != np.float32))[0]] if not(x in self.exclude_cols)]
        self.numerical_columns = [x for x in self.df_train.columns[np.where(np.logical_or(self.df_train.dtypes == np.float64,self.df_train.dtypes == np.float32))[0]] if not(x in self.exclude_cols)]
        
        self.custom_postprocessing_fn = self.kwargs["custom_postprocessing_fn"] if "custom_postprocessing_fn" in kwargs.keys() else None
        
        self.oe = {}
        for c in self.categorical_columns if label_encode else []:
            oe = OrdinalEncoder()
            self.df_train[c] = oe.fit_transform(self.df_train[[c]].values)
            self.oe[c] = oe
        self.df_train_min = self.df_train.min()
        self.df_train_max = self.df_train.max()
        self.ss = {}
        for c in self.categorical_columns+self.numerical_columns if standard_scale_all else (self.numerical_columns if standard_scale else []):
            ss = RobustScaler()
            self.df_train[c] = ss.fit_transform(self.df_train[[c]].values)
            self.ss[c] = ss
            
        self.imputer = None
        
    def impute(self, df_test=None, impute_cols=None, n_imputations=100, return_reduced=True, retrain=False):
        '''
        returns a list of pandas dataframes one for each entry in impute_cols with columns
        id: index
        imputation_id: enumerates the different imputations 0...n_imputations-1
        cols in entry from impute_cols: imputed values (or all values if return_reduced=True)
        optional: sampling_prob: normalized or unnormalized sampling probabilities
        '''
        print(f'Imputing dataset with n = {len(df_test)} samples and {n_imputations} imputations')
        if(impute_cols is None):
            impute_cols = [[x] for x in self.df_train.columns if x not in self.exclude_cols]
        df_test = self.df_train if df_test is None else self.preprocess(df_test.copy())

        if(not(isinstance(impute_cols[0],list)) and not(isinstance(impute_cols[0],np.ndarray))):
            impute_cols = [impute_cols]

        res = self._impute(df_test=df_test, impute_cols=impute_cols, n_imputations=n_imputations, return_reduced=return_reduced, retrain=retrain)
        return [self.postprocess(r) for r in res]

    def _impute(self,  df_test=None, impute_cols=None, n_imputations=100, return_reduced=True, retrain=False):
        '''
        internal imputation routine to be implemented by each specific imputer
        '''
        pass

    def preprocess(self, df):
        '''routine to be applied in derived classes '''
        for c in self.oe.keys():
            df[c] = self.oe[c].transform(df[[c]].values)        
        for c in self.ss.keys():
            df[c] = self.ss[c].transform(df[[c]].values)
        return df

    def postprocess(self, df):
        #round and truncate
        for c in self.ss.keys():
            if(c in df.columns):
                df[c] = self.ss[c].inverse_transform(df[[c]].values)

        for c in self.categorical_columns:
            if(c in df.columns):
                df[c] = df[c].astype(float).round(0).astype(int)      
                df[c] = df[c].clip(lower=self.df_train_min[c], upper=self.df_train_max[c])        
        for c in self.oe.keys():
            if(c in df.columns):
                df[c] = self.oe[c].inverse_transform(df[[c]].values)
        #custom postprocessing
        cols=[x for x in df.columns if x not in ["id","imputation_id","sampling_prob"]]
        if(self.custom_postprocessing_fn is not None):
            df[cols] = self.custom_postprocessing_fn(df[cols])
        return df
    
    
###############################################################################################
# Imputer Implementations
###############################################################################################
class IterativeImputer(ImputerBase):
    '''
    simple imputer based on sklearn's IterativeImputer (uses regressor for all columns)
    
    Parameters:
    n_estimators: number of RF estimators
    max_iter: maximum number of iterations in IterativeImputer
    algorithm: RandomForest/ExtraTrees (standard implementation) or RandomForestSampling/ExtraTreesSampling (with stochastic sampling from a fitted tree)
    '''
    #c.f. https://github.com/scikit-learn/scikit-learn/pull/13025/files
    def __init__(self, df_train, **kwargs):
        super().__init__(df_train, **kwargs)

        self.n_estimators = self.kwargs["n_estimators"] if "n_estimators" in self.kwargs.keys() else 100
        self.max_iter = self.kwargs["n_iter"] if "n_iter" in self.kwargs.keys() else 10
        self.algorithm = self.kwargs["algorithm"] if "algorithm" in self.kwargs.keys() else "RandomForestSampling"
        self.n_jobs = self.kwargs["n_jobs"] if "n_jobs" in self.kwargs.keys() else -1
        if(self.algorithm == "RandomForest"):
            self.cls_regressor = RandomForestRegressor
        elif(self.algorithm == "RandomForestSampling"):
            self.cls_regressor = RandomForestRegressorSampling
        elif(self.algorithm == "ExtraTrees"):
            self.cls_regressor = ExtraTreesRegressor
        elif(self.algorithm == "ExtraTreesSampling"):
            self.cls_regressor = ExtraTreesRegressorSampling
        else:
            assert(False)

    def _impute(self, df_test, impute_cols, n_imputations=100, return_reduced=True, retrain=False):
            
        res=[[] for _ in range(len(impute_cols))]

        include_cols = [x for x in self.df_train.columns if not x in self.exclude_cols]
        integer_cols = self.df_train.columns[np.where(self.df_train.dtypes == np.int64)[0]]
        
        for i in tqdm(list(range(1 if self.algorithm.endswith("Sampling") else  n_imputations))):
            #store one imputer
            if(not(self.algorithm.endswith("Sampling")) or  self.imputer is None or retrain is True):
                self.imputer = IterativeImputerOriginal(self.cls_regressor(n_estimators=self.n_estimators,n_jobs=self.n_jobs),random_state=i, max_iter=self.max_iter, imputation_order="random")
                self.imputer.fit(self.df_train[include_cols])
            else:
                print("Info: using trained imputer; pass retrain=True to retrain") 
            
            for k in tqdm(list(range(n_imputations)),leave=False) if self.algorithm.endswith("Sampling") else [1]:#only fit once for Samplers    
                for j,ic in enumerate(impute_cols):
                    df_test_tmp = df_test[include_cols].copy()            
                    df_test_tmp[ic] = np.nan
                    df_test_tmp = pd.DataFrame(data=self.imputer.transform(df_test_tmp),columns=include_cols)
                    for c in self.exclude_cols:#restore excluded cols
                        df_test_tmp[c]=df_test[c]
                    df_test_tmp["imputation_id"]=k if self.algorithm.endswith("Sampling") else i #add imputation id
                    df_test_tmp["id"]=df_test_tmp.index
                    if(return_reduced):
                        df_test_tmp=df_test_tmp[["id","imputation_id"]+list(ic)]
                    res[j].append(df_test_tmp)
        return [pd.concat(r) for r in res]

class IterativeImputerEnhanced(IterativeImputer):
    '''
    more elaborate imputer that can deal with categorical variables
    c.f. https://github.com/scikit-learn/scikit-learn/pull/13025/files

    Parameters:
    n_estimators: number of RF estimators
    max_iter: maximum number of iterations in IterativeImputer
    algorithm: RandomForest/ExtraTrees (standard implementation) or RandomForestSampling/ExtraTreesSampling (with stochastic sampling from a fitted tree)
    '''
    
    def __init__(self, df_train, **kwargs):
        super().__init__(df_train, **kwargs)

        self.n_estimators = self.kwargs["n_estimators"] if "n_estimators" in self.kwargs.keys() else 100
        self.max_iter = self.kwargs["n_iter"] if "n_iter" in self.kwargs.keys() else 10
        self.algorithm = self.kwargs["algorithm"] if "algorithm" in self.kwargs.keys() else "RandomForestSampling"
        self.n_jobs = self.kwargs["n_jobs"] if "n_jobs" in self.kwargs.keys() else -1
        
        if(self.algorithm == "RandomForest"):
            self.cls_regressor = RandomForestRegressor
            self.cls_classifier = RandomForestClassifier
        elif(self.algorithm == "RandomForestSampling"):
            self.cls_regressor = RandomForestRegressorSampling
            self.cls_classifier = RandomForestClassifierSampling
        elif(self.algorithm == "ExtraTrees"):
            self.cls_regressor = ExtraTreesRegressor
            self.cls_classifier = ExtraTreesClassifier
        elif(self.algorithm == "ExtraTreesSampling"):
            self.cls_regressor = ExtraTreesRegressorSampling
            self.cls_classifier = ExtraTreesClassifierSampling
        else:
            assert(False)

    def _impute(self, df_test, impute_cols, n_imputations=100, return_reduced=True, retrain=False):
            
        res=[[] for _ in range(len(impute_cols))]
        
        impute_cols_flat = np.unique([item for sublist in impute_cols for item in sublist])    
        include_cols = [x for x in self.df_train.columns if not x in self.exclude_cols]
        nonfloat_cols = list(self.df_train.columns[np.where(np.logical_and(self.df_train.dtypes != np.float64,self.df_train.dtypes != np.float32))[0]])
        integer_cols = list(self.df_train.columns[np.where(self.df_train.dtypes == np.int64)[0]])
        classification_cols = [x for x in nonfloat_cols if (x not in self.nocat_cols and x in include_cols)]
        regression_cols = [x for x in include_cols if x not in classification_cols]
        classification_cols_selected = [x for x in classification_cols if x in impute_cols_flat]
        classification_cols_rest = [x for x in classification_cols if not x in impute_cols_flat]
        regression_cols_selected = [x for x in regression_cols if x in impute_cols_flat]
        regression_cols_rest = [x for x in regression_cols if not x in impute_cols_flat]
        
        
        for i in tqdm(list(range(1 if self.algorithm.endswith("Sampling") else n_imputations))):
            #store one imputer for later usage
            if(not(self.algorithm.endswith("Sampling")) or  self.imputer is None or retrain is True):
                self.imputer = IterativeImputerPR(
                    estimator=[
                        (self.cls_classifier(n_estimators=self.n_estimators), classification_cols_selected),
                        (self.cls_regressor(n_estimators=self.n_estimators), regression_cols_selected),
                        (DummyClassifier(strategy="most_frequent"), classification_cols_rest),
                        (DummyRegressor(strategy="median"), regression_cols_rest),

                    ],
                    transformers=[
                        (OneHotEncoder(sparse=False), classification_cols_selected),
                        (StandardScaler(), regression_cols)
                    ],
                    initial_strategy="most_frequent",
                    n_jobs=self.n_jobs, max_iter=self.max_iter,imputation_order="random")

                self.imputer.fit(self.df_train[include_cols])
            else:
                print("Info: using trained imputer; pass retrain=True to retrain") 
            
            for k in tqdm(list(range(n_imputations)),leave=False) if self.algorithm.endswith("Sampling") else [1]:#only fit once for Samplers
                for j,ic in enumerate(impute_cols):
                    df_test_tmp = df_test[include_cols].copy()            
                    df_test_tmp[ic] = np.nan
                    df_test_tmp = pd.DataFrame(data=self.imputer.transform(df_test_tmp),columns=include_cols)
                    for c in self.exclude_cols:#restore excluded cols
                        df_test_tmp[c]=df_test[c]
                    df_test_tmp["imputation_id"]=k if self.algorithm.endswith("Sampling") else i#add imputation id
                    df_test_tmp["id"]=df_test_tmp.index
                    if(return_reduced):
                        df_test_tmp=df_test_tmp[["id","imputation_id"]+list(ic)]
                    res[j].append(df_test_tmp)
        return [pd.concat(r) for r in res]

class TrainSetImputer(ImputerBase):
    '''
    imputer just inserts randomly sampled training samples
    '''
    
    def __init__(self, df_train, **kwargs):
        kwargs["label_encode"]=False
        kwargs["standard_scale"]=False
        super().__init__(df_train, **kwargs)
     
    def _impute(self, df_test, impute_cols, n_imputations=100, return_reduced=True, retrain=False):
            
        res=[[] for _ in range(len(impute_cols))]

        for i in tqdm(list(range(n_imputations))):
            for j, ic in enumerate(impute_cols):
                df_test_tmp = df_test.copy()
                df_test_tmp[ic] = self.df_train[ic].sample(n=len(df_test_tmp)).values
                df_test_tmp["imputation_id"] = i      # add imputation id
                df_test_tmp["id"] = range(len(df_test_tmp))
                if(return_reduced):
                    df_test_tmp = df_test_tmp[["id", "imputation_id"]+list(ic)]
                res[j].append(df_test_tmp)
        return [pd.concat(r) for r in res]

class MedianImputer(ImputerBase):
    '''
    imputer just inserts the median of the training samples (all of them identical so n_imputations should be set to 1)
    '''
    
    def __init__(self, df_train, **kwargs):
        kwargs["label_encode"]=False
        kwargs["standard_scale"]=False
        super().__init__(df_train, **kwargs)
     
    def _impute(self, df_test, impute_cols, n_imputations=100, return_reduced=True, retrain=False):
        res=[[] for _ in range(len(impute_cols))]

        for i in tqdm(list(range(n_imputations))):
            for j,ic in enumerate(impute_cols):
                df_test_tmp = df_test.copy()
                df_test_tmp[ic]=self.df_train[ic].median().values[0]
                df_test_tmp["imputation_id"]=i
                df_test_tmp["id"]=df_test_tmp.index
                if(return_reduced):
                    df_test_tmp=df_test_tmp[["id","imputation_id"]+list(ic)]
                res[j].append(df_test_tmp)

        return [pd.concat(r) for r in res]


def _swap_axis(arr: np.ndarray, axis=Union[int, List[int]], dim: int=None, backward=False) -> np.ndarray:
    """
    iteratively swaps all axis to the end of the matrix, arr.ndim = 1 or 2
    :param arr: square matrix, will be copied
    :param axis: list of integers, will be swaped to the end of the array
    :param dim: defines which dimensions to be swap. If None all dimension will be swapped
    :param backward: perform inverse swap
    :return: permutated matrix
    """
    if isinstance(axis, list) is False:
        assert isinstance(axis, int), f'axis = {axis} of incorrect type.'
        axis = [axis]
    assert arr.ndim == 1 or arr.ndim == 2, 'only vector or two-dimensional matrix allowed'

    target_axis = (arr.shape[0] - np.arange(len(axis)) - 1).tolist()

    dim_one = False
    dim_two = False
    if dim is None:
        dim_one = True
        if arr.ndim == 2:
            dim_two = True
    elif dim == 0:
        dim_one = True
    elif dim == 1:      # only second axis will be swapped
        dim_two = True
        target_axis = (arr.shape[1] - np.arange(len(axis)) - 1).tolist()

    if backward is True:
        axis.reverse()
        target_axis.reverse()

    temp = arr.copy()
    for tar, ax in zip(target_axis, axis):
        if dim_one is True:
            row_org = temp[ax].copy()
            row_target = temp[tar].copy()
            temp[ax], temp[tar] = row_target, row_org      # swap rows
        if dim_two is True:
            col_org = temp[:, ax].copy()
            col_target = temp[:, tar].copy()
            temp[:, ax], temp[:, tar] = col_target, col_org
    return temp


K = np.array([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]])
axis = [0, 3]
a = _swap_axis(K, axis)
b = _swap_axis(a, axis, backward=True)
assert np.alltrue(np.equal(K, b)), 'swap axis function modified'


class GaussianProcessImputer(ImputerBase):
    """
    draws impute samples from a multivariate gaussian. Standard covariance from train samples is used and conditioned on
    """
    
    def __init__(self, df_train, **kwargs):
        kwargs["standard_scale_all"]= True
        super().__init__(df_train, **kwargs)
     
    def _impute(self, df_test, impute_cols, n_imputations=100, return_reduced=True, retrain=False):

        covariance_matrix = self.df_train.cov()
        mean = self.df_train.mean()

        # uncomment to ignore feature correlations
        # covariance_matrix = pd.DataFrame(np.diag(np.diag(covariance_matrix)).reshape(covariance_matrix.shape))

        res = [[] for _ in range(len(impute_cols))]

        for j, ic in tqdm(list(enumerate(impute_cols))):
            for i in tqdm(list(range(n_imputations)), leave=False):
                n_imputed = len(ic)

                # separate training and label features
                index_columns = [df_test.columns.get_loc(key) for key in ic]
                cov_imputed = _swap_axis(covariance_matrix.values, index_columns)
                mean_imputed = _swap_axis(mean.values, index_columns)
                x = _swap_axis(df_test.values, index_columns, dim=1)        # sort only order of feature columns

                mean_train = mean_imputed[:-n_imputed]
                mean_star = mean_imputed[-n_imputed:]

                x_train = x[:, :-n_imputed]
                x_star = x[:, -n_imputed:]

                K_tt = cov_imputed[:-n_imputed, :-n_imputed]    # tt: train, train
                K_st = cov_imputed[-n_imputed:, :-n_imputed]    # st: star/predict, train
                K_ss = cov_imputed[-n_imputed:, -n_imputed:]    # ss: start/predict, star/predict

                temp = scipy.linalg.solve(K_tt, (x_train - mean_train).T)
                mean_conditioned = (mean_star[:, np.newaxis] + K_st @ temp).T
                cov_conditioned = K_ss - K_st @ scipy.linalg.solve(K_tt, K_st.T)

                mvn = scipy.stats.multivariate_normal(mean=np.zeros(n_imputed), cov=cov_conditioned)
                samples = mvn.rvs(x_star.shape[0]).reshape(mean_conditioned.shape) \
                          + mean_conditioned    # n_samples x n_imputed

                # store new samples
                df_test_tmp = df_test.copy()
                df_test_tmp[ic] = samples
                df_test_tmp["imputation_id"] = i  # add imputation id
                df_test_tmp["id"] = df_test_tmp.index
                if return_reduced is True:
                    df_test_tmp = df_test_tmp[["id", "imputation_id"] + list(ic)]
                res[j].append(df_test_tmp)

        return [pd.concat(r) for r in res]

import torch


class TrainSetMahalanobisImputer(ImputerBase):
    """
    implements the Imputer from 1903.10464
    Parameters:
    sigma: occurring in the denominator of the exponential; small sigma- most weight to closest training observations (low bias and high variance); large sigma vice versa
    batch_size_test: process test set in batches
    """
    
    def __init__(self, df_train, **kwargs):
        kwargs["standard_scale_all"]= True
        kwargs["gpus"] = kwargs["gpus"] if "gpus" in kwargs.keys() else 0
        
        super().__init__(df_train, **kwargs)        
        self.sigma = kwargs["sigma"] if "sigma" in kwargs.keys() else 0.1
        self.batch_size_test = kwargs["batch_size_test"] if "batch_size_test" in kwargs.keys() else 0
        
        
    def _impute(self, df_test, impute_cols, n_imputations=100, return_reduced=True, retrain=False):

        non_impute_cols = [[x for x in self.df_train.columns if not(x in self.exclude_cols) and not(x in ic)] for ic in impute_cols]
        train_equals_test = self.df_train.equals(df_test)

        res=[]

        if(self.batch_size_test>0):
            batches = len(df_test)//self.batch_size_test+1 if  len(df_test)%self.batch_size_test >0 else len(df_test)//self.batch_size_test
            batch_id_start = [i*self.batch_size_test for i in range(batches)]
            batch_id_end = [min((i+1)*self.batch_size_test,len(df_test)) for i in range(batches)]
        else:
            batch_id_start = [0]
            batch_id_end = [len(df_test)]


        for j, (ic,nic) in tqdm(list(enumerate(zip(impute_cols,non_impute_cols)))):
            cov = self.df_train[nic].cov()
            covinv = np.linalg.pinv(cov)#was .inv
            df_imputed = []
            for bis,bie in tqdm(list(zip(batch_id_start,batch_id_end)),leave=False):
                xdelta = np.expand_dims(np.array(self.df_train[nic]),1)-np.expand_dims(np.array(df_test[nic].iloc[range(bis,bie)]),0) #trainid,testid,featureid
                ##############
                if(self.kwargs["gpus"]>0):
                    with torch.no_grad():
                        xdelta_torch = torch.from_numpy(xdelta).cuda()
                        covinv_torch = torch.from_numpy(covinv).cuda()
                        distssq = torch.mean(torch.einsum('ijk,kl->ijl',xdelta_torch,covinv_torch)*xdelta_torch,dim=2).cpu().numpy()
                else:
                    distssq = np.mean(np.einsum('ijk,kl->ijl',xdelta,covinv)*xdelta,axis=2) #trainid, testid
                weights = np.exp(-0.5*distssq/self.sigma/self.sigma) #trainid, testid
                #exclude the sample itself if train_equals_test
                train_ids = np.argsort(weights,axis=0)[-n_imputations:,].T if not(train_equals_test) else np.argsort(weights,axis=0)[-n_imputations-1:-1,].T #testid, trainid/n_imputations
                imputation_weights = np.array([weights[sid,i] for i,sid in enumerate(train_ids)]) #testid, n_imputations
                assert np.all(np.sum(imputation_weights,axis=1)>1e-8),"Assert(TrainSetMahalanobisImputer): weights too small. Increase the imputer's sigma parameter."
                imputation_ids = np.repeat([range(n_imputations)],bie-bis,axis=0)
                test_ids = np.array([[i for _ in range(n_imputations)] for i in range(bis,bie)])

                #flatten everything
                train_ids = train_ids.flatten()
                imputation_weights = imputation_weights.flatten()
                imputation_ids = imputation_ids.flatten()
                test_ids = test_ids.flatten()

                if(return_reduced):
                    df_imputed_tmp = self.df_train[ic].iloc[train_ids].copy()
                else:
                    df_imputed_tmp = df_test[ic].iloc[test_ids].copy()
                    df_imputed_tmp[ic] = self.df_train[ic].iloc[train_ids].values
                df_imputed_tmp["id"]=test_ids
                df_imputed_tmp["imputation_id"]=imputation_ids
                df_imputed_tmp["sampling_prob"]=imputation_weights
                df_imputed.append(df_imputed_tmp)
                
            res.append(pd.concat(df_imputed))
        return res