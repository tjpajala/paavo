import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data_transforms
import similarity


def plot_correlations(data):
    sns.set_style('white')
    corr = data.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='Purples', vmax=.8,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def get_pca_cols(data):
    cols = [c for c in data.columns.values if c not in ['pono', 'vuosi', 'nimi', 'pono.level', 'rakennukset_bin',
                                                        'kunta', 'kuntanro', 'pinta_ala', 'geometry', 'posti_alue',
                                                        'posti_aluenro', 'hinta', 'nimi_x', 'vuosi_x', 'naiset_bin']]
    cols.sort()
    return cols


def get_pca_data(data, year, pono_level):
    cols = get_pca_cols(data)
    data_to_pca = data.loc[data["pono.level"] == 5, :]
    # select year
    data_to_pca = data_to_pca.loc[data_to_pca['vuosi'] == year, :]
    if pono_level == 5:
        y = np.array(data_to_pca.loc[:, ['pono']])
        target_names = data_to_pca.loc[:, ['nimi']]['nimi']
        target_names.index = range(len(target_names))
    else:
        y = np.array([format(int(x), '05d') for x in
                      round(data_to_pca.loc[:, ['pono']].astype(int), -5 + pono_level)['pono']])
        # format target names to leading zeros
        target_names = [format(int(x), '05d') for x in
                        round(data_to_pca.loc[:, ['pono']].astype(int), -5 + pono_level)['pono']]

    X = np.array(data_to_pca.loc[:, cols])

    return X, y, target_names


def do_pca(X, n_components):
    pipe = make_pipeline(StandardScaler(), PCA(n_components=n_components))
    X_pca = pipe.fit_transform(X)
    return X_pca, pipe


def pca_2d_plot(X_pca, target_names, color):
    x0 = np.array([(i - min(X_pca[:, 0])) / (max(X_pca[:, 0]) - min(X_pca[:, 0])) for i in X_pca[:, 0]])
    y0 = np.array([(i - min(X_pca[:, 1])) / (max(X_pca[:, 1]) - min(X_pca[:, 1])) for i in X_pca[:, 1]])
    # plt.interactive(False)
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111)
    p = ax.scatter(x=x0, y=y0, cmap='cool', c=color, alpha=0.5, s=8)
    plt.colorbar(p)
    outliers = np.logical_and(get_outliers_bool(x0), get_outliers_bool(y0))
    labels_to_plot = [target_names[i] if outliers[i] is True else '' for i in range(len(target_names))]
    for i in range(len(x0)):
        # plt.annotate(s=re.split(' ', str(labels_to_plot[i]))[0], xy=(x0[i], y0[i]))
        plt.annotate(labels_to_plot[i], xy=(x0[i], y0[i]), alpha=0.7, fontsize='x-large')
    fig.get_axes()[1].set_ylabel('Post code', rotation=270)
    plt.show(block=True)

    return None


def get_outliers_bool(x0):
    return abs(x0 - np.mean(x0)) > 2 * np.std(x0)


def pca_3d_plot(X_pca, target_names, color):
    # Store results of PCA in a data frame
    result = pd.DataFrame(X_pca, columns=['PCA%i' % i for i in range(3)], index=range(len(X_pca)))

    # Plot initialisation
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap='viridis', s=1, alpha=0.8, c=color)
    cbar = plt.colorbar(p)
    plt.show()

    return None


def generate_pca_report(pca):
    print('explained variance ratio: %s' % str(pca.explained_variance_ratio_))
    dims = len(pca.explained_variance_ratio_)
    cols = ['C' + str(i) + '' for i in (range(1, dims + 1))]
    pca_comp = pd.DataFrame(pca.components_.transpose(), columns=cols)

    return pca_comp


def pca_plot(x_pca, target_names, postcodes):
    dims = x_pca.shape[1]
    if dims == 2:
        pca_2d_plot(x_pca, target_names, postcodes)
    elif dims == 3:
        pca_3d_plot(x_pca, target_names, postcodes)
    else:
        print('PCA has %s dims, too many to plot', str(dims))


def exploratory_pca(x, n_components):
    x_ex_pca, pipe_ex = do_pca(x, n_components)
    explained_var = pipe_ex.named_steps['pca'].explained_variance_ratio_
    plt.plot(np.cumsum(np.round(explained_var, decimals=4) * 100))
    plt.axis([1, 20, 0, 100])
    plt.xticks(np.arange(0, 21, 1))
    plt.axhline(y=90, ls='--', c='red')
    plt.grid(True)
    plt.show()

    return None


def missing_plot(data):
    sns.heatmap(data.isnull())
    plt.show()


def table_similar_with_names(data, orig_name, comparison_names, target_names, X_pca, cols, tail=False):
    orig_idx = (target_names == orig_name).idxmax()
    comp_idx = target_names[target_names.isin(comparison_names)].index.tolist()
    all_names = list(comparison_names)
    all_names.append(orig_name)

    if cols is None:
        cols = ['nimi'] + data_transforms.NOMINAL_VARS + ['dist']
    else:
        cols = ['nimi'] + cols + ['dist']
    d = similarity.pairwise_distances(X_pca, X_pca, 'euclidean')
    df = data.copy()
    df['dist'] = d[orig_idx, :]
    df.sort_values(by='dist', ascending=True, inplace=True)
    df = df.loc[df['nimi'].isin(all_names), cols]
    if tail:
        last = df.tail(5)
        df = df.append(last.loc[:, cols])
    df = df.loc[:, cols]
    df = df.iloc[:, ~df.columns.duplicated()]
    return df


def visualize_similar_with_names(data, orig_name, comparison_names, target_names, X_pca, cols_to_plot):
    orig_idx = (target_names == orig_name).idxmax()
    comp_idx = target_names[target_names.isin(comparison_names)].index.tolist()
    all_names = comparison_names.tolist()
    all_names.append(orig_name)
    if cols_to_plot is None:
        cols_to_plot = ['nimi'] + data_transforms.NOMINAL_VARS + ['dist']
    d = similarity.pairwise_distances(X_pca, X_pca, 'euclidean')
    df = data.copy()
    df['dist'] = d[orig_idx, :]
    df.sort_values(by='dist', ascending=True, inplace=True)
    df = df.loc[df['nimi'].isin(all_names), cols_to_plot]
    melted = df.melt(id_vars=['nimi'])

    g = sns.catplot(x="value", y="nimi", col="variable", col_wrap=5, data=melted, margin_titles=True, sharex=False,
                    sharey=True, s=9)

    # Use semantically meaningful titles for the columns
    vartitles = pd.read_csv('paavo_vars.csv', sep=";")
    titles_dict = dict(zip(vartitles['koodi'], vartitles['nimi']))
    titles_dict['dist'] = 'PCA distance'
    titles = [titles_dict.get(x) for x in melted['variable'].unique()]
    for ax, title in zip(g.axes.flat, titles):
        # Set a different title for each axes
        ax.set(title=title)
    #
    #    # Make the grid horizontal instead of vertical
    #    ax.xaxis.grid(False)
    #    ax.yaxis.grid(True)

    # sns.despine(left=True, bottom=True)
    plt.show()

    return df
