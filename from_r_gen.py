import pandas as pd


def load_r_data(filename):
    d2 = pd.read_csv(filename, sep=";")
    d2.drop('Unnamed: 0', axis=1, inplace=True)
    return d2


def get_price_data(data):
    json_folder = '/Users/tjpajala/kannattaako_kauppa/json_2017/predictions/'
    a = pd.DataFrame()
    a['pono'] = data.loc[data['vuosi'] == 2018, 'pono']
    a['price'] = 1e15
    temp = pd.Series(0, index=range(len(a)))
    for row in range(len(a)):
        print(a.iloc[row, 0])
        try:
            res = pd.read_json(json_folder + a.iloc[row, 0] + '.json')
        except ValueError:
            res = pd.DataFrame(data={'year': [2018], 'hinta50': [0]})
        temp[row] = res.loc[res['year'] == 2018, 'hinta50']
    a['price'] = temp
    return a