import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gp
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models.glyphs import Patches

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
                        highlights_idx=None, year=2018, figsize=(16,16)):
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
    df.plot(color='white', edgecolor="grey", alpha=0.1, ax=ax)
    df_origin.plot(color='red', ax=ax)
    df_highlights.plot(color='orange', ax=ax)
    fig.get_axes()[0].set_axis_off()

    plt.annotate(df_origin['nimi_x'].to_string(),
                 xy=(df_origin['geometry'].centroid.x, df_origin['geometry'].centroid.y))
    for index, row in df_highlights.iterrows():
        plt.annotate(row['nimi_x'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y))
    plt.show()


def map_with_highlights_names(dataframe, title='', origin_name=None, highlights=None, year=2018, figsize=(16,16)):
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
    map_with_highlights(dataframe, title, origin_idx, highlights_idx, year, figsize)

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
