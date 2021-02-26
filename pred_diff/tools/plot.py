import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap                 # conda install -c conda-forge shap
import copy

from .. import preddiff
from ..imputers import impute


def add_text(text: str, ax: plt.Axes, loc: str):
    anchored_text = plt.matplotlib.offsetbox.AnchoredText(text,
                                                          loc=loc, borderpad=0.03)
    anchored_text.patch.set_boxstyle("round, pad=-0.2, rounding_size=0.1")
    anchored_text.patch.set_alpha(0.8)
    ax.add_artist(anchored_text)


def plot_performance(reg, x, y):
    plt.figure('performance')
    plt.plot(y, reg.predict(x), '.', label=reg.__module__[-10:])
    diag = np.linspace(y.min(), y.max(), 10)
    plt.plot(diag, diag)
    plt.legend()


def _scatter(c: pd.DataFrame, x_df: pd.DataFrame, method='', error_bars=None):
    plt.figure('Scatter - ' + method)
    columns = np.ceil(np.sqrt(len(c.keys())))
    rows = np.ceil(len(c.keys())/columns)

    for n, key in enumerate(c.keys()):
        ax = plt.subplot(int(rows), int(columns), int(n+1))
        if error_bars is None:
            plt.scatter(x_df[key], c[key], marker='.', s=15)
        if error_bars is not None:
            plt.errorbar(x_df[key], c[key], error_bars[key], marker='.', linestyle='', capsize=1.5, capthick=0.5, s=15)

        character = chr(97+n)      #ASCII character
        add_text(text=f"$x_{character}$", ax=ax, loc='lower right')
        add_text(text=r"${\bar{m}}^{\,\,\mathrm{res}}$", ax=ax, loc='upper left')

        ax.grid(True)

        plt.tight_layout(pad=0.1)


def scatter_2d_heatmap(x: np.ndarray, y: np.ndarray, relevance: np.ndarray, title='', axis_symmetric=False):
    fig = plt.figure(f'heatmap - {title}')

    vmax = 0.8 * np.abs(relevance).max()
    xmax = 1.05*np.abs([x, y]).max()
    xmin = np.array([x, y]).min() - 0.05*xmax
    if axis_symmetric is True:
        xmin = -xmax
    im = plt.scatter(x, y, c=relevance, cmap='coolwarm', vmax=vmax,
               vmin=-vmax, marker='o', alpha=0.9)
    ax = plt.gca()
    add_text(text='$x_a$', ax=ax, loc='lower right')
    add_text(text='$x_b$', ax=ax, loc='upper left')

    plt.axis([xmin, xmax, xmin, xmax])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"interaction  -  ${\bar{m}}^{\,\,\mathrm{int}}$", rotation='vertical')


    fig.tight_layout(pad=0.1)
    plt.tight_layout(pad=0.1)


def scatter_m_plots(reg: RandomForestRegressor, df_train: pd.DataFrame, df_test: pd.DataFrame, imputer, n_imputations,
                    error_bars=False, n_samples=None):
    explainer = preddiff.PredDiff(model=reg, df_train=df_train, imputer_cls=imputer)
    m_list = explainer.relevances(df_test=df_test, n_imputations=n_imputations)

    values = np.array([m["mean"] for m in m_list])
    error = np.array([m["mean"] - m['low'] for m in m_list])
    c = pd.DataFrame(values.T, columns=df_test.columns)
    n_samples = c.shape[0] if n_samples is None else n_samples
    error_bars = pd.DataFrame(error.T, columns=df_test.columns).head(n_samples) if error_bars is True else None
    _scatter(c=c.head(n_samples), x_df=df_test.head(n_samples), method=f'PredDiff, n_imputations = {n_imputations}',
             error_bars=error_bars)


def scatter_contributions(reg: RandomForestRegressor, x_df: pd.DataFrame, method='Sabaas',
                          x_train: pd.DataFrame = None):
    if method == 'TreeExplainer':
        explainer = shap.TreeExplainer(reg)
        contributions = explainer.shap_values(x_df)
    elif method == 'DeepExplainer':
        assert x_train is not None, 'please insert background data x_train'
        # select a set of background examples to take an expectation over
        mask = np.unique(np.random.randint(0, x_train.shape[0], size=100))
        background = x_train.iloc[mask]
        explainer = shap.DeepExplainer(reg, background)
        contributions = explainer.shap_values(x_df)
    elif method == 'KernelExplainer':
        assert x_train is not None, 'please insert background data x_train'
        # select a set of background examples to take an expectation over
        mask = np.unique(np.random.randint(0, x_train.shape[0], size=100))
        background = x_train.iloc[mask]
        ex = shap.KernelExplainer(reg.predict, background)
        contributions = ex.shap_values(x_df)
    else:
        assert 0 == 1, f'method not implemented: {method}'

    c = pd.DataFrame(contributions, columns=x_df.columns)

    _scatter(c=c, x_df=x_df, method=method)


def plot_n_dependence(reg, x_train, x_test, n_imputations):
    n_plot = 100
    data = x_test.iloc[:n_plot]
    imputer = impute.GaussianProcessImputer
    explainer = preddiff.PredDiff(model=reg, df_train=copy.deepcopy(x_train), imputer_cls=imputer)


    relevance_col = [['1']]
    m_list = explainer.relevances(df_test=data, n_imputations=n_imputations, impute_cols=relevance_col)

    values = np.array([m["mean"] for m in m_list])
    error = np.array([m["mean"] - m['low'] for m in m_list])

    figsize = plt.rcParams['figure.figsize'].copy()
    figsize[1] = 0.55*figsize[1]
    title = f'Ndepedence_{n_imputations}'
    fig = plt.figure(title, figsize=figsize)

    key = relevance_col[0]
    ax = fig.add_subplot(1, 2, 1)
    ax.errorbar(np.squeeze(data[key]), np.squeeze(values), np.squeeze(error), marker='.', linestyle='', capsize=1.5, capthick=0.5)
    add_text(text='$x_b$', ax=ax, loc='lower right')
    add_text(text=r'$\bar{m}$', ax=ax, loc='upper left')
    ax.set_yticks([-7, 0, 7])

    data = x_test.iloc[:2*n_plot]
    interaction_cols = [[['0'], ['1']]]
    m_int = explainer.interactions(df_test=data.copy(), interaction_cols=interaction_cols, n_imputations=n_imputations)
    interaction = m_int[0]['mean']

    ax = fig.add_subplot(1, 2, 2)
    axis_symmetric = True
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    vmax = 0.8 * np.abs(interaction).max()
    xmax = 1.05 * np.abs([x, y]).max()
    xmin = np.array([x, y]).min() - 0.05 * xmax
    if axis_symmetric is True:
        xmin = -xmax
    im = ax.scatter(x, y, c=interaction, cmap='coolwarm', vmax=vmax,
                vmin=-vmax, marker='o', alpha=0.8)
    add_text(text='$x_a$', ax=ax, loc='lower right')
    add_text(text='$x_b$', ax=ax, loc='upper left')
    ax.axis([xmin, xmax, xmin, xmax])

    fig.tight_layout(pad=0.1)

