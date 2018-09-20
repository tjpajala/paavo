import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import geopandas as gp
import plotly.graph_objs as go
import plotly.offline as py
import numpy as np
import pickle

import map_fi_plot
import viz
import similarity
import data_transforms

app = dash.Dash()


def make_graph_data(df, fill_color='white', hover=True):
    plot_data4 = []

    for index, row in df.iterrows():
        xs = []
        ys = []
        x_centroids = []
        y_centroids = []
        if row['geometry'].type == 'Polygon':
            #x,y = np.array(row.geometry.exterior.xy)
            x = row.geometry.exterior.xy[0].tolist()
            y = row.geometry.exterior.xy[1].tolist()
            c_x = row.geometry.centroid.xy[0].tolist()
            c_y = row.geometry.centroid.xy[1].tolist()
            #c_x, c_y = np.array(row.geometry.centroid.xy)
            xs = xs + x
            ys = ys + y
            x_centroids = x_centroids + c_x
            y_centroids = y_centroids + c_y
        elif row['geometry'].type == 'MultiPolygon':
            if len(row.geometry)!=1:
                geom=list(row.geometry)
            else:
                geom = list(row.geometry[0])
            #x,y = np.array([poly.exterior.xy for poly in geom])
            #x,y = np.array(row['geometry'].convex_hull.exterior.xy)
            #x = [poly.exterior.xy[0] for poly in geom]
            #y = [poly.exterior.xy[1] for poly in geom]
            x = ([poly.exterior.xy[0].tolist() for
                  poly in row['geometry']])
            y = ([poly.exterior.xy[1].tolist() for
                  poly in row['geometry']])
            #c_x = np.array([poly.centroid.x for poly in geom])
            #c_y = np.array([poly.centroid.y for poly in geom])
            #c_x = np.array(np.mean([poly.centroid.x for poly in geom]))
            #c_y = np.array(np.mean([poly.centroid.y for poly in geom]))
            c_x = [np.mean([poly.centroid.x for poly in geom])]
            c_y = [np.mean([poly.centroid.y for poly in geom])]
            for segment in range(len(x)):
                xs = xs + x[segment]
                ys = ys + y[segment]
                xs.append(np.nan)
                ys.append(np.nan)
            x_centroids = x_centroids + c_x
            y_centroids = y_centroids + c_y
            xs.append(np.nan)
            ys.append(np.nan)

        else:
            print('stop')
        county_outline = dict(
                type = 'scatter',
                showlegend = False,
                legendgroup = "shapes",
                line = dict(color='grey', width=0.2),
                x=xs,
                y=ys,
                fill='toself',
                fillcolor = fill_color,
                mode='lines',
                hoverinfo='none'
        )
        hover_point = dict(
                type = 'scatter',
                showlegend = False,
                legendgroup = "centroids",
                name = '',
                text = row.nimi,
                marker = dict(size=2, color='red', opacity=0),
                x=x_centroids,
                y=y_centroids,
                mode="markers",
                fill='none'
        )
        annotation = dict(
            type = 'scatter',
            showlegend=False,
            x=x_centroids,
            y=y_centroids,
            text='',
            mode='text',
            hoverinfo='none',
            textposition='bottom center'
        )
        plot_data4.append(county_outline)
        plot_data4.append(annotation)
        if hover:
            plot_data4.append(hover_point)


    #fig = dict(data=plot_data4, layout=layout)
    #py.plot(fig, filename='test.html')
    return plot_data4


def define_layout():
    layout = dict(
        hovermode = 'closest',
        xaxis = dict(
            autorange = True,
            #range=[coords_max['minx'], coords_max['maxx']],
            showgrid = True,
            zeroline = False,
            showticklabels=False,
            ticks='outside',
            domain=[0.1, 0.9],
            fixedrange=False
        ),
        yaxis = dict(
            autorange = True,
            #range=[coords_max['miny'], coords_max['maxy']],
            showgrid = True,
            zeroline = False,
            showticklabels=False,
            ticks='outside',
            fixedrange=False,
            scaleanchor='x',
            scaleratio= 1,
            domain=[0.1, 0.9]
        ),
        margin = dict(
            t=0,
            b=0,
            r=0,
            l=0
        ),
        width = 900,
        height = 2*450,
        dragmode = 'select'
    )
    return layout


def format_numeric_table_cols(tb, numcols=None):
    if numcols is None:
        numcols = tb.columns.values[(tb.columns.values != "nimi") & (tb.columns.values != 'pono')]
    tb.loc[:, numcols] = tb.loc[:, numcols].applymap("{0:.2f}".format)
    return tb


df = pd.read_csv('data_to_plotly.csv', dtype={
     'pono': 'O',
     'pono.level': 'int64',
     'vuosi': 'int64',
     'nimi': 'O',
     'he_kika': 'float64',
     'ra_ke': 'float64',
     'ra_raky': 'float64',
     'ra_muut': 'float64',
     'ra_asrak': 'float64',
     'ra_asunn': 'float64',
     'ra_as_kpa': 'float64',
     'ra_pt_as': 'float64',
     'ra_kt_as': 'float64',
     'tp_tyopy': 'float64',
     'tp_alku_a': 'float64',
     'tp_jalo_bf': 'float64',
     'tp_palv_gu': 'float64',
     'tr_mtu': 'float64',
     'te_takk': 'float64',
     'te_as_valj': 'float64',
     'he_naiset': 'float64',
     'hr_hy_tul': 'float64',
     'hr_ke_tul': 'float64',
     'hr_ovy': 'float64',
     'hr_pi_tul': 'float64',
     'pt_0_14': 'float64',
     'pt_elakel': 'float64',
     'pt_opisk': 'float64',
     'pt_tyoll': 'float64',
     'pt_tyott': 'float64',
     'pt_tyovu': 'float64',
     'te_aik': 'float64',
     'te_eil_np': 'float64',
     'te_elak': 'float64',
     'te_laps': 'float64',
     'te_nuor': 'float64',
     'te_omis_as': 'float64',
     'te_vuok_as': 'float64',
     'tr_hy_tul': 'float64',
     'tr_ke_tul': 'float64',
     'tr_pi_tul': 'float64',
     'rakennukset_bin': 'float64',
     'hinta': 'int64',
     'yliopistot': 'int64',
     'amk': 'int64'})

df.drop(labels='Unnamed: 0', axis=1, inplace=True)
df = map_fi_plot.merge_to_polygons_for_year(df, 2018)
#df.drop(labels=['vuosi_y', 'nimi_y'], axis=1, inplace=True)
df = df.rename(index=str, columns={'vuosi_y': 'vuosi', 'nimi_y': 'nimi'})
#get uni data
#amk, yl = data_transforms.get_edu_data()
#df['yliopistot'] = [yl[x] if x in yl else 0 for x in df.pono]
#df['amk'] = [amk[x] if x in amk else 0 for x in df.pono]

X, y, target_names = viz.get_pca_data(df, 2018, 5)
target_names.index = range(len(target_names))
X_pca, pipe = viz.do_pca(X, 5)

#plot_data4 = make_graph_data(df)
#with open('plotly_plot_data4', 'wb') as f:
#    pickle.dump(plot_data4, f)

with open('plotly_plot_data4', 'rb') as f:
    data_pickle = pickle.load(f)


pono_name_dict = dict(zip(df.sort_values(by='pono').pono + ' ' + df.sort_values(by='pono').nimi,
                          df.sort_values(by='pono').nimi))
app.layout = html.Div([
    html.Div([
        html.Div([
            html.P('Select origin area:', style={'display': 'inline-block'}),
            dcc.Dropdown(
                id='origin-area',
                options=[{'label': key, 'value': pono_name_dict.get(key)} for key in pono_name_dict.keys()],
                value=pono_name_dict.get(list(pono_name_dict.keys())[0]),
                searchable=True,
                placeholder='Select your current postcode area'
            ),
            html.P('Select move type:', style={'display': 'inline-block'}),
            dcc.RadioItems(
                id='move-type',
                options=[{'label': 'Move to target', 'value': 'intersection'},
                         {'label': 'Move away from origin', 'value': 'difference'}],
                value='intersection'
            ),
            html.P('Select range in km:', style={'display': 'inline-block'}),
            dcc.Slider(
                id='range-km-slider',
                min=20,
                max=500,
                step=10,
                value=50,
                marks={30: 30,
                       100: 100,
                       150: 150,
                       200: 200,
                       250: 250,
                       300: 300,
                       350: 350,
                       400: 400,
                       450: '450 km'},
                included=True
            ),

            html.P('Select target area:', style={'display': 'inline-block'}),
            dcc.Dropdown(
                id='target-area',
                options=[{'label': key, 'value': pono_name_dict.get(key)} for key in pono_name_dict.keys()],
                value=pono_name_dict.get(list(pono_name_dict.keys())[0]),
                searchable=True,
                placeholder='Select your target postcode area'
            ),
            html.P('What price per sq. meter can you accept at max?', style={'display': 'inline-block'}),
            dcc.Slider(
                id='price-slider',
                min=df['hinta'].min(),
                max=df['hinta'].max(),
                value=df['hinta'].max(),
                step=1000,
                marks={0: 0,
                       1000: '1000',
                       2000: '2000',
                       3000: '3000',
                       4000: '4000',
                       5000: '5000 €'
                       },
                included=True
            ),
            html.P('How many alternatives do you want: ', style={'display': 'inline-block'}),
            dcc.Input(
                id='nmost-input',
                inputmode='numeric',
                min=3,
                max=30,
                step=1,
                type='number',
                value=10
            ),
            dcc.Graph(id='comparison-table')
            ],
            style={'width': '35%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='indicator-graphic'),

        ], style={'width': '65%', 'float': 'right', 'display': 'inline-block'})
    ], style={'display': 'flex', 'align-items': 'flex-start', 'justify-content': 'space-around'}),




], )
@app.callback(
    dash.dependencies.Output('comparison-table', 'figure'),
    [dash.dependencies.Input('origin-area', 'value'),
     dash.dependencies.Input('move-type', 'value'),
     dash.dependencies.Input('range-km-slider', 'value'),
     dash.dependencies.Input('nmost-input', 'value'),
     dash.dependencies.Input('price-slider', 'value'),
     dash.dependencies.Input('target-area', 'value')])
def update_table(origin_name, move_type,
                 range_km, n_most,
                 max_price, target):
    df_filtered = similarity.filter_w_price(df, max_price, [origin_name, target])
    #transform values to ranks for easy understanding
    df_filtered_ranks = similarity.full_df_to_ranks(df_filtered, bins=10)
    df_origin = df.loc[df['nimi'] == origin_name, :]
    if move_type == 'difference':
        target = origin_name
    area, included = map_fi_plot.get_included_area(df_filtered, move_type, origin_name, range_km, target)
    NA, included_ranks = map_fi_plot.get_included_area(df_filtered_ranks, move_type, origin_name, range_km, target)
    X, y, target_names = viz.get_pca_data(included, 2018, 5)
    target_names.index = range(len(target_names))
    X_pca = pipe.transform(X)
    d = similarity.pairwise_distances(X_pca, X_pca, 'euclidean')
    similar = similarity.get_similar_in_geo_area(included, origin_name, d,
                                                 target_names, n_most)
    tb = viz.table_similar_with_names(included_ranks, origin_name, similar, target_names, X_pca, ['pono','nimi','he_kika',
                                                                                            'ra_asunn','te_laps',
                                                                                            'te_as_valj','tp_tyopy',
                                                                                            'tr_mtu', 'yliopistot', 'amk'],
                                      tail=False)
    tb = tb.drop_duplicates()
    
    tb = format_numeric_table_cols(tb, numcols=['dist'])
    cols = [x for x in tb.columns.values if x not in ['geometry', 'kunta', 'kuntanro', 'pono', 'pono.level', 'nimi', 'nimi_x', 'vuosi',
                          'dist', 'rakennukset_bin']]
    #tb.loc[:, cols] = tb.loc[:, cols].applymap(lambda x: similarity.value_to_plusses(x))

    trace = go.Table(
        header=dict(values=list(['Pono','Nimi','Keski-ikä', 'Asunnot', 'Lapsitaloudet', 'Työpaikat','Mediaanitulo', 'Yliopistot', 'AMK' ,'Dist']),
                    fill=dict(color='#C2D4FF'),
                    align=['left'] * 5,
                    height=40),
        cells=dict(values=[tb.pono, tb.nimi, tb.he_kika, tb.ra_asunn, tb.te_laps, tb.tp_tyopy, tb.tr_mtu, tb.yliopistot, tb.amk, tb.dist],
                   fill=dict(color='#F5F8FF'),
                   align=['left'] * 5,
                   height=30)
    )
    #py.plot([trace], 'test.html')
    return {'data': [trace],
            'layout': dict(autosize=True, margin=dict(
                               t=0,
                               b=0,
                               r=0,
                               l=0
                           )
                           )
            }

    # return html.Table(
    #     # Header
    #     [html.Tr([html.Th(col) for col in tb.columns])] +
    #
    #     # Body
    #     [html.Tr([
    #         html.Td(tb.iloc[i][col]) for col in tb.columns
    #     ]) for i in range(min(len(tb), len(tb)))]
    # )



@app.callback(
    dash.dependencies.Output('indicator-graphic', 'figure'),
    [dash.dependencies.Input('origin-area', 'value'),
     dash.dependencies.Input('move-type', 'value'),
     dash.dependencies.Input('range-km-slider', 'value'),
     dash.dependencies.Input('nmost-input', 'value'),
     dash.dependencies.Input('price-slider', 'value'),
     dash.dependencies.Input('target-area', 'value')])
def update_graph(origin_name, move_type,
                 range_km, n_most,
                 max_price, target):
    df_filtered = similarity.filter_w_price(df, max_price, [origin_name, target])
    df_origin = df.loc[df['nimi'] == origin_name, :]
    if move_type == 'difference':
        target=origin_name
    area, included = map_fi_plot.get_included_area(df_filtered, move_type, origin_name, range_km, target)
    X, y, target_names = viz.get_pca_data(included, 2018, 5)
    target_names.index = range(len(target_names))
    X_pca = pipe.transform(X)
    d = similarity.pairwise_distances(X_pca, X_pca, 'euclidean')
    similar = similarity.get_similar_in_geo_area(included, origin_name, d,
                                                 target_names, n_most)
    df_comparison = df.loc[df['nimi'].isin(similar), :]
    coords_max = {
        'miny': df.bounds.miny.min(),
        'minx': df.bounds.minx.min(),
        'maxy': df.bounds.miny.max(),
        'maxx': df.bounds.miny.max()
    }

    layout = define_layout()

    return {
#        'data': set_fill_colors_for_origin_and_comp(plot_data4, origin_name, similar, target),
        'data': set_fill_colors_for_origin_and_comp(make_graph_data(included), origin_name, similar, target),
        'layout':  layout
    }


def set_fill_colors_for_origin_and_comp(plot_data4, origin_name, similar, target):
    #reset colors to white
    all_idx = range(0, len(plot_data4), 3)
    for idx in all_idx:
        plot_data4[idx]['fillcolor'] = 'white'
        plot_data4[idx+1]['text'] = ''
    origin_idx = [x for x in range(2, len(plot_data4), 3) if plot_data4[x]['text']==origin_name][0]
    plot_data4[origin_idx-2]['fillcolor'] = 'red'
    plot_data4[origin_idx-1]['text'] = plot_data4[origin_idx]['text']
    comp_idx = [x for x in range(2, len(plot_data4), 3) if plot_data4[x]['text'] in similar]
    for idx in comp_idx:
        plot_data4[idx-2]['fillcolor'] = 'orange'
        plot_data4[idx - 1]['text'] = plot_data4[idx]['text']
    target_idx = [x for x in range(2, len(plot_data4), 3) if plot_data4[x]['text']==target][0]
    plot_data4[target_idx - 2]['fillcolor'] = 'purple'
    plot_data4[target_idx - 1]['text'] = 'Target: '+plot_data4[target_idx]['text']
    return plot_data4


if __name__ == '__main__':
    app.run_server()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
