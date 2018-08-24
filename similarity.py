import numpy as np
import sklearn


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
