import load_data
import pandas as pd
import numpy as np
import data_transforms
import importlib
import viz
from itertools import chain
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import from_r_gen
import similarity
import map_fi_plot
from persons import engineer_w_kid, student_21F
importlib.reload(from_r_gen)
importlib.reload(load_data)
importlib.reload(data_transforms)
importlib.reload(viz)
importlib.reload(similarity)
importlib.reload(map_fi_plot)

data = from_r_gen.load_r_data('paavo_counts.csv')
data2 = from_r_gen.load_r_data('paavo_shares.csv')

#combine nominal and share vars
data = data.loc[:, list(chain.from_iterable([['pono', 'pono.level', 'vuosi', 'nimi'], data_transforms.NOMINAL_VARS]))]
data2 = data2.loc[:, list(chain.from_iterable([['pono', 'vuosi', 'nimi'], data_transforms.SHARES_VARS]))]


cols_to_use = data2.columns.difference(data.columns)
data = pd.merge(data, data2[cols_to_use], left_index=True, right_index=True, how='outer')
data = data.reindex()

data.loc[data['pono.level'] == 5, 'pono'] = [format(x, '05d') for x in data.loc[data['pono.level'] == 5, 'pono']]
data.loc[data['pono.level'] == 3, 'pono'] = [format(x*100, '05d') for x in data.loc[data['pono.level'] == 3, 'pono']]
data.loc[data['pono.level'] == 2, 'pono'] = [format(x*100, '05d') for x in data.loc[data['pono.level'] == 2, 'pono']]


to_format = ['he_kika', 'tr_mtu', 'ra_as_kpa', 'hr_pi_tul', 'hr_ke_tul', 'hr_hy_tul',
             'pt_tyoll', 'pt_tyott', 'pt_tyovu', 'pt_0_14', 'pt_opisk', 'pt_elakel', 'hr_ovy',
             'tr_pi_tul', 'tr_ke_tul', 'tr_hy_tul', 'te_nuor', 'te_eil_np', 'te_laps', 'te_aik',
             'te_elak', 'te_omis_as', 'te_vuok_as', 'te_takk', 'te_as_valj']
for c in to_format:
    data[c] = [float(str(x).replace(",", ".")) for x in data[c]]


#save aggregated to different df
data_agg = data.loc[data["pono.level"] != 5, :]
data = data.loc[data['pono.level'] == 5, :]

viz.missing_plot(data)
data.fillna(0, inplace=True)
#data.fillna(data.mean(), inplace=True)
viz.missing_plot(data)

#select only one year
data = data.loc[data['vuosi'] == 2018, :]

X, y, target_names = viz.get_pca_data(data, 2018, 5)
target_names.index = range(len(target_names))
viz.exploratory_pca(X, 20)


X_pca, pipe = viz.do_pca(X, 5)

pca_comp = viz.generate_pca_report(pipe.named_steps['pca'])
pca_comp['vars'] = viz.get_pca_cols(data)
print(pca_comp)

viz.pca_plot(X_pca, target_names, y.ravel())
#save pca to csv
#pca_comp.to_csv('pca.csv')

#similarity calculation
d = similarity.pairwise_distances(X_pca, X_pca, 'euclidean')
names = similarity.get_n_most_similar_with_name("Otaniemi", d, target_names, 10)
print(names)

data_l5 = data.loc[data['pono.level'] == 5, :].assign(max_factor=pd.DataFrame(X_pca.argmax(axis=1)))
map_fi_plot.map_fi_postinumero(data_l5, "Highest factors per area", color_var='max_factor')

map_fi_plot.map_with_highlights_names(data_l5, "How similar to Jupperi?", 'Jupperi',
                                      similarity.get_n_most_similar_with_name('Jupperi', d, target_names, 15))
map_fi_plot.map_with_highlights_names(data_l5, "How similar to Otaniemi?", 'Otaniemi',
                                      similarity.get_n_most_similar_with_name('Otaniemi', d, target_names, 15))
names = similarity.get_n_most_similar_with_name("Otaniemi", d, target_names, 10)
print(names)
viz.visualize_similar_with_names(data, orig_name='Otaniemi', comparison_names=names,
                                 target_names=target_names, X_pca=X_pca, cols_to_plot=None)

#test person to plot where he should move
places_most_similar = similarity.get_n_most_similar_to_person_with_names(engineer_w_kid, pipe, X_pca, 15, target_names)
print(places_most_similar)
map_fi_plot.map_with_highlights_names(data_l5, "Where should an 29M engineer with kids move?",
                                      origin_name='Möckelö', highlights=places_most_similar)
viz.visualize_similar_with_names(data_l5, 'Möckelö', places_most_similar, target_names, X_pca, cols_to_plot=None)
#similarity.write_similar_to_csv(places_most_similar, engineer_w_kid, data_l5, 'compare_engineer.csv')



places_most_similar = similarity.get_n_most_similar_to_person_with_names(student_21F, pipe, X_pca, 15, target_names)
print(places_most_similar)
map_fi_plot.map_with_highlights_names(data_l5, "Where should a 21F student with kids move?",
                                      origin_name='Möckelö', highlights=places_most_similar)

#similarity.write_similar_to_csv(places_most_similar, student_21F, data_l5, 'compare_student21F.csv')


viz.visualize_similar_with_names(data, orig_name='Otaniemi', comparison_names=names,
                                 target_names=target_names, X_pca=X_pca, cols_to_plot=None)


