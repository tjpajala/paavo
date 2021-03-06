from itertools import chain

import numpy as np
import pandas as pd
import sklearn.metrics


def pairwise_distances(x, y, metric):
    return sklearn.metrics.pairwise_distances(x, y, metric)


def name_to_matrix_row(target_names, name_to_search):
    if target_names[target_names == name_to_search].any(None):
        return target_names[target_names == name_to_search].index[0]
    else:
        raise ValueError('Name does not exist in target_names!')


def get_n_most_similar(mat, index, n_most):
    mat_internal = np.copy(mat)
    np.fill_diagonal(mat_internal, 1e15)
    return np.argpartition(mat_internal[index, :], range(n_most))[0:n_most]


def get_n_most_similar_with_name(origin_name, mat, target_names, n_most):
    row_to_use = name_to_matrix_row(target_names, origin_name)
    most_similar = get_n_most_similar(mat, row_to_use, n_most)
    return target_names[most_similar]


def write_similar_to_csv(places_most_similar, person, data_l5, filename):
    places = data_l5.loc[data_l5['nimi'].isin(places_most_similar.values), :]
    append_student = pd.DataFrame([list(chain.from_iterable([['00000', 0, 2018, 'Student'], person, [0]]))],
                                  columns=places.columns)
    compare = places.append(append_student)
    compare.to_csv(filename)


def get_similar_in_geo_area(included_area, orig_name, d, target_names, n_most):
    similar = get_n_most_similar_with_name(orig_name, d, target_names, len(target_names) - 1).tolist()
    similar = [x for x in similar if x in included_area.nimi.tolist()][0:n_most]
    return similar


def filter_w_price(data, max_price, including_names=[]):
    return data.loc[(data['hinta'] <= max_price) | (data['nimi'].isin(including_names)), :]


def variable_to_ranks(df, col_name, bins=5):
    return pd.cut(df.loc[:, col_name].rank(), bins=bins, labels=range(1, bins + 1))


def full_df_to_ranks(df, bins=5):
    data = df.copy()
    nonnumeric_columns = ['geometry', 'kunta', 'kuntanro', 'pono', 'pono.level', 'nimi', 'nimi_x', 'vuosi',
                          'dist', 'rakennukset_bin']
    for col in data.columns:
        if col not in nonnumeric_columns:
            data.loc[:, col] = variable_to_ranks(data, col, bins=bins)
    return data


def value_to_plusses(rank):
    return rank * '*'


def table_by_group(data, grouping_var, variable, metric):
    if metric not in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        raise ValueError("Incorrect argument metric, acceptable values are: count, mean, std, min, 25%, 50%, 75%, max")
    return data.groupby(by=grouping_var).describe().loc[:, variable].loc[:, metric]
