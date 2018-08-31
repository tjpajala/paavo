import numpy as np
import sklearn
import pandas as pd
from itertools import chain


def similarity_for_two_objects(func, x, y):
    return func(x, y)


def similarity_for_matrix(func, x, mat):
    return [similarity_for_two_objects(func, x, mat[i, :]) for i in range(0, len(mat))]


def euclidean(x, y):
    return np.linalg.norm(x-y)


def pairwise_distances(X, Y, metric):
    return sklearn.metrics.pairwise_distances(X, Y, metric)


def get_closest_index(mat, index):
    mat_internal = np.copy(mat)
    np.fill_diagonal(mat_internal, 1e15)
    return np.argmin(mat_internal[index, :])


def name_to_matrix_row(target_names, name_to_search):
    if target_names[target_names == name_to_search].any(None):
        return target_names[target_names == name_to_search].index[0]
    else:
        raise ValueError('Name does not exist in target_names!')


def get_n_most_similar(mat, index, n_most):
    mat_internal = np.copy(mat)
    np.fill_diagonal(mat_internal, 1e15)
    return np.argpartition(mat_internal[index, :], n_most)[0:n_most]


def get_n_most_similar_with_name(origin_name, mat, target_names, n_most):
    row_to_use = name_to_matrix_row(target_names, origin_name)
    most_similar = get_n_most_similar(mat, row_to_use, n_most)
    return target_names[most_similar]


def get_n_most_similar_to_person(person, pipe, X_pca, n_most, metric=None):
    #fill missing values with data means
    person_fill = np.array(pd.Series(person).fillna(pd.Series(pipe.named_steps['standardscaler'].mean_))).reshape(1, -1)
    #scale and project vector with previous pipeline
    person_pca = pipe.transform(person_fill)
    pca_data = np.insert(X_pca, 0, person_pca, axis=0)
    if metric is None:
        metric = 'euclidean'
    d = pairwise_distances(pca_data, pca_data, metric)
    return get_n_most_similar(d, 0, n_most)


def get_n_most_similar_to_person_with_names(person, pipe, X_pca, n_most, target_names, metric=None):
    if metric is None:
        metric = 'euclidean'
    similar = get_n_most_similar_to_person(person, pipe, X_pca, n_most, metric)
    return target_names[similar]

def write_similar_to_csv(places_most_similar, person, data_l5, filename):
    places = data_l5.loc[data_l5['nimi'].isin(places_most_similar.values), :]
    append_student = pd.DataFrame([list(chain.from_iterable([['00000', 0, 2018, 'Student'], person, [0]]))],
                                  columns=places.columns)
    compare = places.append(append_student)
    compare.to_csv(filename)