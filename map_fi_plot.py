import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import similarity
import viz
from bokeh.models import GeoJSONDataSource
from bokeh.models.glyphs import Patches
from bokeh.plotting import figure, output_file, show

CITIES = [
    'Helsinki Keskusta - Etu-Töölö',
    'Tampere Keskus',
    'Oulu Keskus',
    'Turku Keskus',
    'Kuopio Keskus',
    'Joensuu Keskus Eteläinen',
    'Seinäjoki Keskus',
    'Rovaniemi Keskus',
    'Vaasa Keskus',
    'Jyväskylä Keskus',
    'Pori Keskus',
    'Lahti Asemanseutu',
    'Kouvola Keskus',
    'Lappeenranta keskus',
    'Hämeenlinna Keskus',
    'Mikkeli Keskus',
    'Utsjoki Keskus',
    'Kittilä Keskus',
    'Sodankylä Keskus',
    'Kokkola Keskus',
    'Pyhäntä Keskus',
    'Savonlinna Keskus',
    'Kajaani Keskus'
]


def merge_to_polygons_for_year(dataframe, year):
    pol = gp.GeoDataFrame.from_file('pno_' + str(year) + '.shp')
    pol["posti_aluenro"] = pd.to_numeric(pol["posti_alue"])
    df = pol.loc[pol['vuosi'] == year, ['posti_alue', 'nimi', 'kunta',
                                        'kuntanro', 'vuosi', 'pinta_ala',
                                        'geometry', 'posti_aluenro']].merge(
        dataframe[dataframe['vuosi'] == year], left_on="posti_alue", right_on="pono")
    return df


def map_fi_postinumero(dataframe, title='', color_var='pt_tyoll', year=2018, cmap='summer', plot_cities=True):

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    plt.title(title)
    df = merge_to_polygons_for_year(dataframe, year)
    df.plot(column=color_var, cmap=cmap, ax=ax, legend=True)
    if plot_cities:
        for index, row in df.loc[df['nimi_x'].isin(CITIES), :].iterrows():
            plt.scatter(x=row['geometry'].centroid.x, y=row['geometry'].centroid.y, color='red', s=3)
            plt.annotate(row['nimi_x'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y), color='red')
    fig.get_axes()[0].set_axis_off()
    fig.get_axes()[1].set_ylabel(color_var, rotation=270)
    plt.show()


def map_with_highlights(dataframe, title='', origin_idx=None,
                        highlights_idx=None, year=2018, figsize=(16,16), area=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.title(title)
    df = merge_to_polygons_for_year(dataframe, year)
    if origin_idx is None:
        origin_idx = np.random.choice(range(len(df)), size=1, replace=False).min()
    else:
        origin_idx = df.loc[df['posti_alue'] == dataframe.iloc[origin_idx, :]['pono'], :].index.tolist()[0]
    df_origin = df.iloc[[origin_idx], :]  #pass list to retain it as dataframe
    if highlights_idx is None:
        highlights_idx = np.random.choice(range(len(df)), size=15, replace=False)
    else:
        highlights_idx = df.loc[df['posti_alue'].isin(dataframe.iloc[highlights_idx, :]['pono']), :].index.tolist()
    df_highlights = df.iloc[highlights_idx, :]
    df.plot(facecolor='none', edgecolor="grey", alpha=0.1, ax=ax)
    if area is not None:
        area.plot(facecolor='none', edgecolor='purple', alpha=0.3, ax=ax)
    df_origin.plot(facecolor='red', ax=ax)
    df_highlights.plot(facecolor='orange', ax=ax)
    fig.get_axes()[0].set_axis_off()

    plt.annotate(df_origin['nimi_x'].to_string(),
                 xy=(df_origin['geometry'].centroid.x, df_origin['geometry'].centroid.y))
    for index, row in df_highlights.iterrows():
        plt.annotate(row['nimi_x'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y))
    plt.show()


def map_with_highlights_names(dataframe, title='', origin_name=None, highlights=None, year=2018, figsize=(16,16), area=None):
    df = merge_to_polygons_for_year(dataframe, year)
    if origin_name not in list(df['nimi_x']):
        raise ValueError('origin_name not in data!')
    if sum([x in list(df['nimi_x']) for x in highlights]) != len(highlights):
        raise ValueError('one or more highlights not in data!')
    if origin_name is None:
        origin_name = df['nimi_x'].get(np.random.choice(range(len(df['nimi_x'])), size=1, replace=False).min())
    if highlights is None:
        highlights = [df['nimi_x'].get(x) for x in np.random.choice(range(len(df['nimi_x'])), size=15, replace=False)]
    origin_idx = (dataframe['nimi'] == origin_name).idxmax()
    highlights_idx = dataframe.index[dataframe['nimi'].isin(highlights)].tolist()
    map_with_highlights(dataframe, title, origin_idx, highlights_idx, year, figsize, area)


def bokeh_map(dataframe, title='', origin_name=None, highlights=None, year=2018):

    output_file('test.html')
    df = merge_to_polygons_for_year(dataframe,year)

    TOOLTIPS = [
        ("Postinumero:", "$pono"),
        ("(x,y)", "($euref_x, $euref_y)"),
    ]

    p = figure(plot_width=600, plot_height=1000, tooltips=TOOLTIPS,
               title="Mouse over the dots")

    df_origin = df.loc[df['nimi_x']==origin_name, :]
    df_highlights = df.loc[df['nimi_x'].isin(highlights), :]

    glyph_orig = Patches(xs="xs", ys="ys", fill_color="red")
    glyph_comp = Patches(xs="xs", ys="ys", fill_color="orig")

    p.multi_line('xs', 'ys', source=GeoJSONDataSource(geojson=df.to_json()), color='gray', line_width=0.5, alpha=0.7)
    p.add_glyph(GeoJSONDataSource(geojson=df_origin.to_json()), glyph_orig)
    p.add_glyph(GeoJSONDataSource(geojson=df_highlights.to_json()), glyph_comp)

    show(p)


def plot_similar_in_geo_area(data, orig_name, target, range_km, how, n_most, pipe):
    methods = ['intersection', 'difference']
    if how not in methods:
        raise ValueError('how should be either "intersection" or "difference"')
    if target is None:
        target = orig_name
    df = merge_to_polygons_for_year(data, 2018)
    if orig_name not in list(df['nimi_x']):
        raise ValueError('origin_name not in data!')
    if target not in list(df['nimi_x']):
        raise ValueError('target not in data!')
    #range expressed in kms
    #limit in shapely units
    limit = range_km*1000
    area = gp.GeoDataFrame()
    area['geometry'] = df.loc[df['nimi_x'] == target, 'geometry'].buffer(limit)
    area.crs = df.crs
    included = gp.overlay(df, area, how=how)
    included = included.append(df.loc[df['nimi_x']==orig_name, :], sort=True)
    included.drop(labels=['vuosi_y', 'nimi_y', 'posti_alue', 'posti_aluenro'], axis=1, inplace=True)
    included.rename(index=str, columns={'posti_alue':'pono','nimi_x':'nimi', 'vuosi_x':'vuosi'}, inplace=True)
    X, y, target_names = viz.get_pca_data(included, 2018, 5)
    target_names.index = range(len(target_names))
    X_pca = pipe.transform(X)
    d = similarity.pairwise_distances(X_pca, X_pca, 'euclidean')
    s2 = similarity.get_n_most_similar_with_name(orig_name, d, target_names, n_most)
    #idx = target_names.isin(included['nimi_x'].append(pd.Series(orig_name)))
    similar = similarity.get_similar_in_geo_area(included, orig_name, d,
                                                 target_names, n_most)
    #included.plot(alpha=0.5, edgecolor='k', cmap='tab10')
    map_with_highlights_names(data, '', orig_name, similar, 2018, area=area)

