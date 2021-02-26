import numpy as np
import pandas as pd
import scipy
from typing import List
from tqdm.auto import tqdm, trange

from .imputers import impute
from .tools.utils_bootstrap import empirical_bootstrap
from .tools import utils_preddiff as ut_preddiff
from functools import partial

global pbar

#####################################################################################################
# Main Class
#####################################################################################################
class PredDiff(object):
    def __init__(self, model, df_train: pd.DataFrame, regression=True, imputer_cls=impute.IterativeImputer,
                 classifier_fn_call="predict_proba", regressor_fn_call="predict",
                 fast_evaluation=False, n_group=1, **imputer_kwargs):
        """
        Arguments:
        model: trained regressor/classifier
        df_train: dataframe for imputer training
        regression: regression or classification setting (classification return relevance for each class)
        classification_logprob: use the logprob rather than the output probabilities to calculate m-values
        imputer_cls: imputer class from impute.py
        classifier_fn_call: member function to be called to evaluate classifier object model
                            (sklearn default: predict_proba)- has to return prediction as numpy array
        regressor_fn_call: member function to be called to evaluate regressor object model
                            (sklearn default: predict)- has to return predictions as numpy array
        imputer_kwargs: other arguments to be passed to imputer
        fast_evaluation: default: False, useful to speed-up evaluation, ignores bootstrapping and returns zero-errorbars
        """
        self.model = model
        self.df_train = df_train
        self.n_samples_train = len(self.df_train)
        self.imputer_cls = imputer_cls
        self.imputer_kwargs = imputer_kwargs
        self.fast_evaluation = fast_evaluation

        self.mangage_cols = ut_preddiff.ManageCols(columns=self.df_train.columns)
        self.imputer = self.imputer_cls(df_train=self.df_train, **self.imputer_kwargs)

        # customize for regression and classification
        self.regression = regression
        self.fn_call = regressor_fn_call if regression else classifier_fn_call
        # where to apply log_transform for classification
        self.log_transform = self.regression is False

        self.model_predict = getattr(self.model, self.fn_call)
        self.n_classes = self.model_predict(df_train.iloc[:2]).shape[1] if self.regression is False else None

        self.n_group = n_group
        self.pbar = None

    def relevances(self,  df_test, impute_cols=None, n_imputations=100, n_bootstrap=100, retrain=False)\
            -> List[pd.DataFrame]:
        """
        computes relevances from a given PredDiff object
        params:
        df_test: dataset for which the relevances are supposed to be calculated
        impute_col: list of list with columns for which the relevance is to be calculated
        n_impuations: number of impuation samples to be used
        n_bootstrap: number of bootstrap samples for confidence intervals, set zero for ignoring and speed up
        retrain: enforce retraining of the imputer
        """
        print('Calculate PredDiff relevances')

        if impute_cols is None:
            impute_cols = self.mangage_cols.relevance_default_cols()
        if not (isinstance(impute_cols[0], list)) and not (isinstance(impute_cols[0], np.ndarray)):
            impute_cols = [impute_cols]

        # get imputations
        list_imputed_values, df_test = self._setup_imputation(df_test, impute_cols, n_imputations, retrain)

        # get predictions
        predictions_true, predictions_raw, shape_imputations = self._setup_predict(df_test, n_imputations)

        print("Evaluating m-values for every element in impute_cols...")
        m_list = []
        self.pbar = tqdm(total=len(list_imputed_values), desc='Relevance: ')

        for imputed_df, cols in zip(list_imputed_values, impute_cols):

            predictions, weights = self._calc_predictions(df_test=df_test, imputed_df=imputed_df,
                                                          shape_imputations=shape_imputations, cols=cols)

            self.pbar.set_postfix_str(f'Evaluate m-values')
            # evaluate and calculate m-values
            eval_mean = partial(
                ut_preddiff.eval_mean_relevance, log_transform=self.log_transform,
                n_train_samples=self.n_samples_train, n_classes=self.n_classes)

            mean, low, high, _ = empirical_bootstrap(
                input_tuple=predictions if weights is None else (predictions, weights),
                score_fn=eval_mean, n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)


            mean = predictions_true - mean
            lowtmp = predictions_true - high    # swap low and high here
            high = predictions_true - low
            low = lowtmp

            m_df = pd.DataFrame([{"mean": m, "low": l, "high": h, "pred": p}
                                 for m, l, h, p in zip(mean, low, high, predictions_raw)])
            m_list.append(m_df)
            self.pbar.set_postfix_str(f'finished column')
            self.pbar.update(1)

        self.pbar.close()
        return m_list

    def interactions(self, df_test, interaction_cols=None, n_imputations=100, n_bootstrap=100,
                     individual_contributions=False, retrain=False) -> List[pd.DataFrame]:
        """
        computes relevances from a given PredDiff object
        params:
        df_test: dataset for which the relevances are supposed to be calculated
        interaction_cols: list of lists of lists [ [[x],[y]]],[[x],[y,z]] ]
                i.e. 1st component: interaction between col x and col y  and
                    2nd component: interaction between col x and [cols y and z]
        n_impuations: number of impuation samples to be used
        n_bootstrap: number of bootstrap samples for confidence intervals
        individual_contributions: also return individual components whose sum makes up the interaction relevance
        retrain: enforce retraining of the imputer
        """
        print('Calculate PredDiff interactions.')
        if interaction_cols is None:
            interaction_cols = self.mangage_cols.interaction_default_cols()

        impute_cols = [[*i1, *i2] for [i1, i2] in interaction_cols]      # join interaction_cols for combined imputation

        list_imputed_values, df_test = self._setup_imputation(df_test, impute_cols, n_imputations, retrain)

        predictions_true, predictions_raw, shape_imputations = self._setup_predict(df_test, n_imputations)

        assert len(impute_cols) == len(list_imputed_values)
        print("Evaluating m-values for every element in interaction_cols...")
        self.pbar = tqdm(total=len(list_imputed_values), desc='Interaction: ')
        m_list = []
        for imputed_df, cols, icols in zip(list_imputed_values, impute_cols, interaction_cols):
            # infer predictions with imputed test samples
            predictions01, weights = self._calc_predictions(df_test=df_test, imputed_df=imputed_df,
                                                            shape_imputations=shape_imputations, cols=cols)
            predictions0, weights = self._calc_predictions(df_test=df_test, imputed_df=imputed_df,
                                                           shape_imputations=shape_imputations, cols=icols[0])
            predictions1, weights = self._calc_predictions(df_test=df_test, imputed_df=imputed_df,
                                                           shape_imputations=shape_imputations, cols=icols[1])

            self.pbar.set_postfix_str(f'evaluate m-values')
            # evaluate m-values
            input_tuple = (predictions01, predictions0, predictions1) if weights is None \
                else (predictions01, predictions0, predictions1, weights)
            eval_interaction = partial(ut_preddiff.eval_mean_interaction,
                                       log_transform=self.log_transform, n_train_samples=self.n_samples_train,
                                       n_classes=self.n_classes)

            mean, low, high, _ = empirical_bootstrap(input_tuple=input_tuple, score_fn=eval_interaction,
                                                     n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)
            mean += predictions_true
            low += predictions_true
            high += predictions_true

            if individual_contributions is True:
                eval_mean = partial(ut_preddiff.eval_mean_relevance, weights=weights, log_transform=self.log_transform,
                                    n_train_samples=self.n_samples_train, n_classes=self.n_classes)

                mean01 = predictions_true - eval_mean(predictions01)
                mean0 = predictions_true - eval_mean(predictions0)
                mean1 = predictions_true - eval_mean(predictions1)
                res = [{'mean': m, 'high': h, 'low': l, 'mean01': m01, 'mean0': m0, 'mean1': m1}
                       for m, h, l, m01, m0, m1 in zip(mean, high, low, mean01, mean0, mean1)]
            else:
                res = [{'mean': m, 'high': h, 'low': l} for m, h, l in zip(mean, high, low)]

            m_df = pd.DataFrame(res)
            m_list.append(m_df)
            self.pbar.update(1)
        self.pbar.close()
        return m_list

    def convert_to_mean_array(self, m_list: List[pd.DataFrame]) -> np.ndarray:
        """
        extracts mean from relevance list and returns an array
        :param m_list: list of pd.DataFrame relevances
        :return: m_array: shape = (n_classes, n_samples, n_features), for regression n_classes is suppressed
        """
        mean_list = [m_feature['mean'] for m_feature in m_list]
        if self.regression is False:
            mean_list_array = []
            for m_feature in mean_list:
                tmp = np.array(list(m_feature))
                mean_list_array.append(tmp)
            m_array = np.swapaxes(np.array(mean_list_array), 0, 2)          # (n_classes, n_samples, n_features)
        else:
            m_array = np.array(mean_list).T             # (n_samples, n_features)
        return m_array

    def interaction_matrix(self, df_test=None, n_imputations=100, n_bootstrap=100) -> np.ndarray:
        """
        returns np.array containing all feature interaction for each sample.
        :param df_test: if given, all samples are interpreted, pd.DataFrame
        :param n_imputations: used for statistics
        :param n_bootstrap:
        :return: m_matrix: shape = (n_samples, n_features, n_features), a relevance for all interaction
        """
        self.fast_evaluation = True     # ignore statistical errors

        # single feature and interaction relevance
        m_list_interaction = self.interactions(df_test=df_test, n_imputations=n_imputations)
        m_list_relevance = self.relevances(n_imputations=n_imputations, df_test=df_test)  # all features relevances

        relevance_array = self.convert_to_mean_array(m_list_relevance)
        interaction_array = self.convert_to_mean_array(m_list_interaction)

        # get n_classes, for regression add extra dimension with n_classes=1
        if self.regression is True:
            relevance_array = relevance_array[np.newaxis]
            interaction_array = interaction_array[np.newaxis]
        n_classes = relevance_array.shape[0]
        assert n_classes == interaction_array.shape[0]

        list_m_matrix = []
        for n in range(n_classes):
            m_array = interaction_array[n]
            # create a lower/upper triangular form out of m_array => create symmetric interaction array
            m_matrix = np.array([scipy.spatial.distance.squareform([x for x in m_array[k, :]])
                                 for k in np.arange(m_array.shape[0])])

            # calculate self-interaction
            assert np.diagonal(m_matrix, axis1=1, axis2=2).sum() == 0, 'non-zero elements on diagonal'
            for i in range(relevance_array.shape[-1]):
                m_matrix[:, i, i] = relevance_array[n, :, i] - m_matrix[:, :, i].sum(axis=1)

            list_m_matrix.append(m_matrix)

        return np.array(list_m_matrix).squeeze()

    def _setup_predict(self, df_test: pd.DataFrame, n_imputations: int):
        n_samples = len(df_test)

        predictions = self.model_predict(df_test)
        predictions_raw = predictions.copy()

        # prellocate memory for all imputations
        if self.regression is False:
            assert predictions.shape[1] == self.n_classes, 'something went wrong during prediction'
            shape_imputations = (n_imputations, n_samples, self.n_classes)
            predictions = ut_preddiff.log_laplace(predictions, self.n_samples_train, self.n_classes)
        else:
            shape_imputations = (n_imputations, n_samples)
        return predictions, predictions_raw, shape_imputations

    def _setup_imputation(self, df_test: pd.DataFrame, impute_cols, n_imputations: int, retrain: bool):
        df_test = self.df_train.copy() if df_test is None else df_test.copy()
        list_imputed_values = self.imputer.impute(df_test=df_test, impute_cols=impute_cols, n_imputations=n_imputations,
                                                  return_reduced=True, retrain=retrain)
        assert len(impute_cols) == len(list_imputed_values)
        return list_imputed_values, df_test

    def _calc_predictions(self, df_test: pd.DataFrame, imputed_df: pd.DataFrame, shape_imputations,
                          cols):
        # prepare variables
        predictions = np.zeros(shape_imputations)
        n_imputations = shape_imputations[0]
        n_samples = len(df_test)
        if self.n_group > n_imputations:
            self.n_group = n_imputations

        assert (n_imputations % self.n_group == 0), \
            f'Speed up factor {self.n_group = } does not match {n_imputations = }\n' \
            f'{n_imputations % self.n_group = } not zero\n' \
            f'Maybe try {np.gcd(n_imputations, int(n_imputations/2)) = }'

        # convert df into more handy format
        index_batch = [i for i in range(self.n_group)]
        df_test_imputed = pd.concat([df_test.copy() for _ in range(self.n_group)], keys=index_batch)

        imputed_grp = imputed_df.groupby('imputation_id')
        imputatations = pd.concat([data[cols] for name, data in imputed_grp], keys=[i for i in range(n_imputations)])

        if 'sampling_prob' in imputed_df.columns:
            weights = np.array([imputed_grp.get_group(i)['sampling_prob'] for i in range(n_imputations)])
            assert np.all(np.sum(weights, axis=1) > 1e-8), \
                "Assert(PredDiff): weights too small. Adjust imputer parameters."
            if self.regression is False:
                weights = np.expand_dims(weights, axis=2)
        else:
            weights = None

        # loop grouped imputation batches
        for i in range(n_imputations)[::self.n_group]:
            self.pbar.set_postfix_str(f'impute [{i}, {i+self.n_group}]')
            # TODO: custom imputation function for different data modalities
            df_test_imputed[cols] = imputatations.loc[i:i + self.n_group - 1][cols].values

            shape = (self.n_group, n_samples) if self.n_classes is None else (self.n_group, n_samples, self.n_classes)
            predictions[i:i + self.n_group] = self.model_predict(df_test_imputed).reshape(shape)

        return predictions, weights
