import numpy as np


#####################################################################################################
# AUX FUNCTIONS
#####################################################################################################
def laplace_smoothing(p, N, d, alpha=1):
    p_laplace = (p*N+alpha)/(N+d)
    return p_laplace


def log_laplace(p, N, d, alpha=1):
    return np.log2(laplace_smoothing(p, N, d, alpha=alpha))


def eval_mean_relevance(predictions, weights=None, log_transform=False, n_train_samples=10000, n_classes = 10):
    if weights is None:
        mean = predictions.mean(axis=0)
    else:
        mean = np.sum(predictions*weights, axis=0)/np.sum(weights, axis=0)
    if log_transform:
        mean = log_laplace(mean, n_train_samples, n_classes)
    return mean


def eval_mean_interaction(predictions01, predictions0, predictions1, weights=None, log_transform=False,
                          n_train_samples=10000, n_classes=10):
    if weights is not None:
        mean01 = np.sum(predictions01*weights, axis=0)/np.sum(weights, axis=0)
        mean0 = np.sum(predictions0*weights, axis=0)/np.sum(weights, axis=0)
        mean1 = np.sum(predictions1*weights, axis=0)/np.sum(weights, axis=0)
    else:
        mean01 = predictions01.mean(axis=0)
        mean0 = predictions0.mean(axis=0)
        mean1 = predictions1.mean(axis=0)
    if log_transform is True:
        mean01 = log_laplace(mean01, n_train_samples, n_classes)
        mean0 = log_laplace(mean0, n_train_samples, n_classes)
        mean1 = log_laplace(mean1, n_train_samples, n_classes)
    return mean01 - mean0 - mean1


class ManageCols:
    def __init__(self, columns):
        """
        This class is meant to be the interface between the special PredDiff columns interface and other data modalities
        such as two-dimensional image data.
        """
        self.columns = columns

    def relevance_default_cols(self):
        """
        generates a list containing all features
        """
        impute_cols = [[x] for x in self.columns]
        return impute_cols

    def interaction_default_cols(self):
        """
        generates a list of all feature combinations
        """
        interaction_cols = []
        for i, k1 in enumerate(self.columns):
            for k2 in self.columns[i + 1:]:
                interaction_cols.append([[k1], [k2]])
        return interaction_cols
