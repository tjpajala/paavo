#load data from stats.fi
#dataframes = load_data.load_all_data(year='2015')

#csv_files = glob.glob(os.getcwd()+'/paavo_[0-9]*.csv')
#df = load_data.combine_csv_files(csv_files)
#df.to_csv("paavo_full.csv")

df = pd.read_csv('paavo_full.csv')
whole_finland = df.loc[df["Postinumeroalue"] == "00000 KOKO MAA"]
df = df.loc[df["Postinumeroalue"] != "00000 KOKO MAA"]
print(len(df.columns.values))

SELECT_COLS = []

years_in_columns = ['2015', '2014', '2013']
for y in years_in_columns:
    df = data_transforms.divide_by_inhabitants(df, y)
    df = data_transforms.divide_by_households(df, y)
    SELECT_COLS.append(['Asukkaat yhteensä, ' + y + ' (PT)'])
    SELECT_COLS.append(['Taloudet yhteensä, ' + y + ' (TR)'])
    SELECT_COLS.append([s+', percent' for s in data_transforms.get_cols_to_divide_by_households(y)])
    SELECT_COLS.append([s + ', percent' for s in data_transforms.get_cols_to_divide_by_inhabitants(y)])
    SELECT_COLS.append(data_transforms.get_nominal_cols(y))
#print(df.columns.values)


A = list(chain.from_iterable(SELECT_COLS))
A.insert(0, "Postinumeroalue")
print(A)

data = df.loc[:, A].fillna(0).reset_index(drop=True)


y = data.index.values
X = data.loc[:, data.columns != "Postinumeroalue"]
target_names = data["Postinumeroalue"]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))


lw = 2
#plot separately only for one year - correlations are similar for all years!
data_subset_cols_nominal = list(chain.from_iterable([data_transforms.get_nominal_cols(y) for y in ['2015']]))
viz.plot_correlations(data.loc[:, data_subset_cols_nominal])

data_subset_cols_inhabitants = list([y+', percent' for y in data_transforms.get_cols_to_divide_by_inhabitants('2015')] +
                                    list(['Asukkaat yhteensä, 2015 (PT)'])) + list(['Taloudet yhteensä, 2015 (TR)'])
viz.plot_correlations(data.loc[:, data_subset_cols_inhabitants])

data_subset_cols_households = list([y+', percent' for y in data_transforms.get_cols_to_divide_by_households('2015')] +
                                    list(['Asukkaat yhteensä, 2015 (PT)'])) + list(['Taloudet yhteensä, 2015 (TR)'])
viz.plot_correlations(data.loc[:, data_subset_cols_households])
