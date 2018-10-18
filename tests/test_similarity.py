import similarity as sim
import numpy as np
import pandas as pd
import pytest
import viz


@pytest.fixture
def get_mock_dataframe():
    def _get_mock_dataframe(last_varname):
        return pd.DataFrame(
            {"nimi": ["nimi_a", "nimi_b", "joku_muu"],
             "koko": [1, 2, 3],
             last_varname: [100, 100, 200]})

    return _get_mock_dataframe


@pytest.fixture()
def get_sample_geodata():
    df = pd.read_csv('./tests/test_data.csv', index_col=False).drop('Unnamed: 0', axis=1)
    return df


def test_pairwise_distance():
    assert (sim.pairwise_distances(np.array([1, 0]).reshape(1, -1), np.array([1, 0]).reshape(1, -1), "euclidean") == 0)
    assert sim.pairwise_distances(np.array([2, 0]).reshape(1, -1), np.array([1, 0]).reshape(1, -1), "euclidean") == 1


def test_name_to_matrix_row(get_mock_dataframe):
    df = get_mock_dataframe(last_varname="asukkaat")
    assert sim.name_to_matrix_row(target_names=df.loc[:, "nimi"], name_to_search="nimi_b") == 1
    with pytest.raises(ValueError) as e:
        sim.name_to_matrix_row(df.loc[:, "nimi"], "does_not_exist")
        assert "Name does not exist in target_names!" in str(e.value)


def test_get_n_most_similar():
    mat = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    assert list(sim.get_n_most_similar(mat, 0, 1)) == list([1])
    assert list(sim.get_n_most_similar(mat, 2, 2)) == list(np.array([0, 1]))


def test_get_n_most_similar_with_name(get_mock_dataframe):
    df = get_mock_dataframe(last_varname="asukkaat")
    origin_name = "nimi_a"
    mat = df.loc[:, ["koko", "asukkaat"]]
    target_names = df.nimi
    n_most = 1
    assert sim.get_n_most_similar_with_name(origin_name, mat, target_names, n_most).tolist() == ['nimi_b']


def test_get_similar_in_geo_area(get_sample_geodata):
    included_area = get_sample_geodata.loc[get_sample_geodata.pono.isin(['00180', '00200', '00210'])]
    x, y, target_names = viz.get_pca_data(get_sample_geodata, 2018, 5)
    target_names.index = range(len(target_names))
    x_pca, pipe = viz.do_pca(x, 5)
    d = sim.pairwise_distances(x_pca, x_pca, 'euclidean')
    res = sim.get_similar_in_geo_area(included_area, orig_name="Vattuniemi", d=d, target_names=target_names, n_most=1)
    assert res == ['Lauttasaari']
    included_area2 = get_sample_geodata.loc[get_sample_geodata.pono.isin(['00180', '00210'])]
    assert sim.get_similar_in_geo_area(included_area2, orig_name="Vattuniemi", d=d, target_names=target_names,
                                       n_most=1) == ["Kamppi - Ruoholahti"]


def test_filter_w_price(get_mock_dataframe):
    df = get_mock_dataframe("hinta")
    assert (df.loc[df.hinta < 101, :] == sim.filter_w_price(df, max_price=100)).all().all()
    assert (df.loc[df.hinta < 201, :] == sim.filter_w_price(df, max_price=200)).all().all()
    assert (df.loc[df.hinta < 201, :] == sim.filter_w_price(df, max_price=100,
                                                            including_names=['joku_muu'])).all().all()


def test_variable_to_ranks(get_mock_dataframe):
    df = get_mock_dataframe("hinta")
    assert (sim.variable_to_ranks(df, "koko", bins=3).values == [1, 2, 3]).all()
    assert (sim.variable_to_ranks(df, "hinta", bins=2).values == [1, 1, 2]).all()


def test_full_df_to_ranks(get_mock_dataframe):
    df = get_mock_dataframe("hinta")
    df_res = pd.DataFrame(
        {"nimi": ["nimi_a", "nimi_b", "joku_muu"],
         "koko": [1, 2, 3],
         "hinta": [1, 1, 3]}
    )
    assert (sim.full_df_to_ranks(df, bins=3) == df_res).all().all()


def test_value_to_plusses():
    assert sim.value_to_plusses(5) == '*****'
    assert sim.value_to_plusses(1) == '*'
