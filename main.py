import importlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import data_transforms
import from_r_gen
import load_data
import map_fi_plot
import similarity
import viz

importlib.reload(from_r_gen)
importlib.reload(load_data)
importlib.reload(data_transforms)
importlib.reload(viz)
importlib.reload(similarity)
importlib.reload(map_fi_plot)

data = load_data.load_plotly_ready_data()
data.drop(labels=["nimi_x","vuosi_x", "geometry", "posti_alue", "posti_aluenro",
                  "kunta","kuntanro","pinta_ala"], axis=1, inplace=True)
# data = from_r_gen.load_r_data('paavo_counts.csv')
# data2 = from_r_gen.load_r_data('paavo_shares.csv')
#
# data = data_transforms.merge_and_clean_data(data, data2)
#
# # save aggregated to different df
# data_agg = data.loc[data["pono.level"] != 5, :]
# data = data.loc[data['pono.level'] == 5, :]
#
# # set yliopistot and amk
# amk, yl = data_transforms.get_edu_data()
# data['yliopistot'] = [yl[x] if x in yl else 0 for x in data.pono]
# data['amk'] = [amk[x] if x in amk else 0 for x in data.pono]

# set rakennukset_bin again, so that it is more fine-grained
data = data_transforms.cut_to_bins(data, 'ra_asrak', 'rakennukset_bin')

if data.isnull().apply(sum,1).sum() != 0:
    viz.missing_plot(data)
    for column_to_impute in data.columns.values:
        if column_to_impute not in ['pono', 'pono.level', 'vuosi', 'nimi']:
            data.loc[data[column_to_impute].isnull(), column_to_impute] = data_transforms. \
                impute_with_class_mean(data, column_to_impute, based_on='rakennukset_bin')

# does mean exist for each variable and class?
# data_transforms.check_class_means(data)
viz.missing_plot(data)

# select only one year
data = data.loc[data['vuosi'] == 2018, :]


# get price data from kannattaako_kauppa
if 'hinta' not in data.columns.values:
    a = from_r_gen.get_price_data(data)
    if np.all(data['pono'] == a['pono']):
        data['hinta'] = a['price']

X, y, target_names = viz.get_pca_data(data, 2018, 5)
target_names.index = range(len(target_names))
viz.exploratory_pca(X, 20)

X_pca, pipe = viz.do_pca(X, 5)

pca_comp = viz.generate_pca_report(pipe.named_steps['pca'])
pca_comp['vars'] = viz.get_pca_cols(data)
print(pca_comp)

viz.pca_plot(X_pca, target_names, y.ravel())
# save pca to csv
# pca_comp.to_csv('pca.csv')

# similarity calculation
d = similarity.pairwise_distances(X_pca, X_pca, 'euclidean')
names = similarity.get_n_most_similar_with_name("Otaniemi", d, target_names, 10)
print(names)

data.reset_index(inplace=True,drop=True)
data_l5 = data.loc[data['pono.level'] == 5, :].assign(max_factor=pd.DataFrame(X_pca.argmax(axis=1)))
map_fi_plot.map_fi_postinumero(data_l5, "Highest factors per area", color_var='max_factor')

map_fi_plot.map_with_highlights_names(data_l5, "How similar to Vattuniemi?", 'Vattuniemi',
                                      similarity.get_n_most_similar_with_name('Vattuniemi', d, target_names, 15))
map_fi_plot.map_with_highlights_names(data_l5, "How similar to Otaniemi?", 'Otaniemi',
                                      similarity.get_n_most_similar_with_name('Otaniemi', d, target_names, 10))
names = similarity.get_n_most_similar_with_name("Vattuniemi", d, target_names, 10)
print(names)
viz.visualize_similar_with_names(data, orig_name='Vattuniemi', comparison_names=names,
                                 target_names=target_names, X_pca=X_pca, cols_to_plot=None)

viz.visualize_similar_with_names(data, orig_name='Otaniemi', comparison_names=names,
                                 target_names=target_names, X_pca=X_pca, cols_to_plot=None)

map_fi_plot.plot_similar_in_geo_area(data, orig_name='Vattuniemi', target='Jyväskylä Keskus', range_km=100,
                                     how='intersection', n_most=15, pipe=pipe)

map_fi_plot.plot_similar_in_geo_area(data, orig_name='Otaniemi', target='Otaniemi', range_km=100,
                                     how='difference', n_most=15, pipe=pipe)

# with price filter
map_fi_plot.plot_similar_in_geo_area(similarity.filter_w_price(data, 4000, including_names=['Vattuniemi']),
                                     orig_name='Vattuniemi', target='Vattuniemi', range_km=100,
                                     how='difference', n_most=15, pipe=pipe)

cols = [x for x in data.columns.values if
        x not in ['geometry', 'kunta', 'kuntanro', 'pono', 'pono.level', 'nimi', 'nimi_x', 'vuosi',
                  'dist', 'rakennukset_bin']]

data = data_transforms.cut_to_bins(data, 'he_naiset', 'naiset_bin')
similarity.table_by_group(data, grouping_var="naiset_bin", variable="amk", metric="mean")

for c in cols:
    data.loc[:, c].plot(kind="hist", title=c)
    plt.show()
