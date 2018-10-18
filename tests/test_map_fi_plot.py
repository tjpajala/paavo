import map_fi_plot
import numpy as np
import pandas as pd
import pytest
import geopandas as gp
import shapely.wkt

@pytest.fixture()
def get_sample_geodata():
    get_sample_geodata = pd.read_csv('./tests/test_data.csv', index_col=False).drop('Unnamed: 0', axis=1)
    return get_sample_geodata


def test_merge_to_polygons_for_year(get_sample_geodata):
    get_sample_geodata.loc[:,"pono"]=[str(x).rjust(5,'0') for x in get_sample_geodata.pono]
    res=map_fi_plot.merge_to_polygons_for_year(get_sample_geodata,2018).iloc[0:2,0:5]
    df_res = pd.DataFrame({"posti_alue_x": ['00310', '00690'],
                           "nimi_x": ['Kivihaka', 'Tuomarinkylä-Torpparinmäki'],
                           "kunta_x": ['091', '091'],
                           "kuntanro_x": [91.0, 91.0],
                           "vuosi_x": [2018.0, 2018.0]},
                          index=[0, 1])
    assert (res == df_res).all().all()

@pytest.mark.skip(reason="no way of currently testing this")
def test_map_fi_postinumero():
    assert False

@pytest.mark.skip(reason="no way of currently testing this")
def test_map_with_highlights():
    assert False

@pytest.mark.skip(reason="no way of currently testing this")
def test_map_with_highlights_names():
    assert False

@pytest.mark.skip(reason="no way of currently testing this")
def test_bokeh_map():
    assert False

@pytest.mark.skip(reason="no way of currently testing this")
def test_plot_similar_in_geo_area():
    assert False


def test_get_included_area(get_sample_geodata):
    df = gp.GeoDataFrame(get_sample_geodata)
    df[['geometry']] = df.geometry.apply(shapely.wkt.loads)
    area, included = map_fi_plot.get_included_area(df, how="intersection", orig_name="Vattuniemi",range_km=0, target="Vattuniemi")
    included.drop_duplicates(subset="pono", inplace=True)
    included.loc[:, "pono"]=[str(x).rjust(5,'0') for x in included.pono]
    assert included.pono.values=="00210"
    area, included = map_fi_plot.get_included_area(df, how="difference", orig_name="Vattuniemi", range_km=10, target="Vattuniemi")
    included.drop_duplicates(subset="pono", inplace=True)
    included.loc[:, "pono"] = [str(x).rjust(5, '0') for x in included.pono]
    array_res = np.array([
        '00690', '00390', '00410', '00420', '00430', '00560', '00680',
        '00700', '00710', '00720', '00730', '00740', '00800', '00750',
        '00760', '00770', '00780', '00790', '00910', '00820', '00920',
        '00930', '00830', '00940', '00950', '00960', '00840', '00850',
        '00860', '00880', '00970', '00980', '00890', '00900', '00990',
        '00670', '00640', '00650', '00660', '00210'], dtype=object)
    assert (included.pono.values == array_res).all()