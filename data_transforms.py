import re
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import chain


def get_inhabitants(df, year):
    col = re.sub('year', year, 'Asukkaat yhteensä, year (HE)')
    return df[col]


def get_cols_to_divide_by_inhabitants(year):
    cols_to_divide_by_inh = [
        'Alimpaan tuloluokkaan kuuluvat asukkaat, year (HR)',
        'Keskimmäiseen tuloluokkaan kuuluvat asukkaat, year (HR)',
        'Ylimpään tuloluokkaan kuuluvat asukkaat, year (HR)',
        # 'Työvoima, year (PT)',
        'Työlliset, year (PT)',
        'Työttömät, year (PT)',
        'Työvoiman ulkopuolella olevat, year (PT)',
        'Lapset 0-14 -vuotiaat, year (PT)',
        'Opiskelijat, year (PT)',
        'Eläkeläiset, year (PT)',
        # 'Muut, year (PT)',
        'Naiset, year (HE)',
        # 'Miehet, year (HE)',
        # 'Perusasteen suorittaneet, year (KO)', #missing for 2015
        # 'Koulutetut yhteensä, year (KO)', #missing for 2015
        # 'Ylioppilastutkinnon suorittaneet, year (KO)', #missing for 2015
        # 'Ammatillisen tutkinnon suorittaneet, year (KO)',
        # 'Alemman korkeakoulututkinnon suorittaneet, year (KO)',
        # 'Ylemmän korkeakoulututkinnon suorittaneet, year (KO)',
        'Asukkaiden ostovoimakertymä, year (HR)'

    ]
    cols_to_divide_by_inh = [re.sub('year', year, s) for s in cols_to_divide_by_inh]
    return cols_to_divide_by_inh


def divide_by_inhabitants(df, year):
    cols = get_cols_to_divide_by_inhabitants(year)
    inhabitants = get_inhabitants(df, year)
    for c in cols:
        df[c + ', percent'] = df[c] / inhabitants
    return df


def get_households(df, year):
    col = re.sub('year', year, 'Taloudet yhteensä, year (TR)')
    return df[col]


def get_cols_to_divide_by_households(year):
    cols_to_divide_by_households = [
        'Alimpaan tuloluokkaan kuuluvat taloudet, year (TR)',
        'Keskimmäiseen tuloluokkaan kuuluvat taloudet, year (TR)',
        'Ylimpään tuloluokkaan kuuluvat taloudet, year (TR)',
        # 'Talouksien ostovoimakertymä, year (TR)',
        'Nuorten yksinasuvien taloudet, year (TE)',
        'Lapsettomat nuorten parien taloudet, year (TE)',
        'Lapsitaloudet, year (TE)',
        # 'Pienten lasten taloudet, year (TE)',
        # 'Alle kouluikäisten lasten taloudet, year (TE)',
        # 'Kouluikäisten lasten taloudet, year (TE)',
        # 'Teini-ikäisten lasten taloudet, year (TE)',
        'Aikuisten taloudet, year (TE)',
        'Eläkeläisten taloudet, year (TE)',
        'Omistusasunnoissa asuvat taloudet, year (TE)',
        'Vuokra-asunnoissa asuvat taloudet, year (TE)',
        # 'Muissa asunnoissa asuvat taloudet, year (TE)'
    ]
    cols_to_divide_by_households = [re.sub('year', year, s) for s in cols_to_divide_by_households]
    return cols_to_divide_by_households


def divide_by_households(df, year):
    cols = get_cols_to_divide_by_households(year)
    households = get_households(df, year)
    for c in cols:
        df[c + ', percent'] = df[c] / households
    return df


NOMINAL_COLUMNS = [
    'Asukkaiden keski-ikä, year (HE)',
    'Kesämökit yhteensä, year (RA)',
    'Rakennukset yhteensä, year (RA)',
    'Muut rakennukset yhteensä, year (RA)',
    'Asuinrakennukset yhteensä, year (RA)',
    'Asunnot, year (RA)',
    'Asuntojen keskipinta-ala, year (RA)',
    'Pientaloasunnot, year (RA)',
    'Kerrostaloasunnot, year (RA)',
    'Työpaikat yhteensä, year (TP)',
    'Alkutuotannon työpaikat, year (TP)',
    'Jalostuksen työpaikat, year (TP)',
    'Palveluiden työpaikat, year (TP)',
    # 'Talouksien keskitulot, year (TR)',
    'Talouksien mediaanitulot, year (TR)',
    'Talouksien keskikoko, year (TE)',
    'Asumisväljyys, year (TE)'
]

NOMINAL_VARS = [
    # 'Asukkaiden keski-ikä, year (HE)',
    'he_kika',
    # 'Kesämökit yhteensä, year (RA)',
    'ra_ke',
    # 'Rakennukset yhteensä, year (RA)',
    'ra_raky',
    # 'Muut rakennukset yhteensä, year (RA)',
    'ra_muut',
    # 'Asuinrakennukset yhteensä, year (RA)',
    'ra_asrak',
    # 'Asunnot, year (RA)',
    'ra_asunn',
    # 'Asuntojen keskipinta-ala, year (RA)',
    'ra_as_kpa',
    # 'Pientaloasunnot, year (RA)',
    'ra_pt_as',
    # 'Kerrostaloasunnot, year (RA)',
    'ra_kt_as',
    # 'Työpaikat yhteensä, year (TP)',
    'tp_tyopy',
    # 'Alkutuotannon työpaikat, year (TP)',
    'tp_alku_a',
    # 'Jalostuksen työpaikat, year (TP)',
    'tp_jalo_bf',
    # 'Palveluiden työpaikat, year (TP)',
    'tp_palv_gu',
    # 'Talouksien keskitulot, year (TR)',
    # 'tr_ktu',
    # 'Talouksien mediaanitulot, year (TR)',
    'tr_mtu',
    # 'Talouksien keskikoko, year (TE)',
    'te_takk',
    # 'Asumisväljyys, year (TE)'
    'te_as_valj'
]

SHARES_VARS = [
    # 'Alimpaan tuloluokkaan kuuluvat asukkaat, year (HR)',
    'hr_pi_tul',
    # 'Keskimmäiseen tuloluokkaan kuuluvat asukkaat, year (HR)',
    'hr_ke_tul',
    # 'Ylimpään tuloluokkaan kuuluvat asukkaat, year (HR)',
    'hr_hy_tul',
    # 'Työvoima, year (PT)',
    # 'pt_tyovy',
    # 'Työlliset, year (PT)',
    'pt_tyoll',
    # 'Työttömät, year (PT)',
    'pt_tyott',
    # 'Työvoiman ulkopuolella olevat, year (PT)',
    'pt_tyovu',
    # 'Lapset 0-14 -vuotiaat, year (PT)',
    'pt_0_14',
    # 'Opiskelijat, year (PT)',
    'pt_opisk',
    # 'Eläkeläiset, year (PT)',
    'pt_elakel',
    # 'Muut, year (PT)',
    # 'pt_muut',
    # 'Naiset, year (HE)',
    'he_naiset',
    # 'Miehet, year (HE)',
    # 'he_miehet',
    # 'Perusasteen suorittaneet, year (KO)', #missing for 2015
    # 'ko_perus',
    # 'Koulutetut yhteensä, year (KO)', #missing for 2015
    # 'ko_koul',
    # 'Ylioppilastutkinnon suorittaneet, year (KO)', #missing for 2015
    # 'ko_yliop',
    # 'Ammatillisen tutkinnon suorittaneet, year (KO)',
    # 'ko_ammat',
    # 'Alemman korkeakoulututkinnon suorittaneet, year (KO)',
    # 'ko_al_kork',
    # 'Ylemmän korkeakoulututkinnon suorittaneet, year (KO)',
    # 'ko_yl_kork',
    # 'Asukkaiden ostovoimakertymä, year (HR)',
    'hr_ovy',
    # 'Alimpaan tuloluokkaan kuuluvat taloudet, year (TR)',
    'tr_pi_tul',
    # 'Keskimmäiseen tuloluokkaan kuuluvat taloudet, year (TR)',
    'tr_ke_tul',
    # 'Ylimpään tuloluokkaan kuuluvat taloudet, year (TR)',
    'tr_hy_tul',
    # 'Talouksien ostovoimakertymä, year (TR)',
    # 'tr_ovy',
    # 'Nuorten yksinasuvien taloudet, year (TE)',
    'te_nuor',
    # 'Lapsettomat nuorten parien taloudet, year (TE)',
    'te_eil_np',
    # 'Lapsitaloudet, year (TE)',
    'te_laps',
    # 'Pienten lasten taloudet, year (TE)',
    # 'te_plap',
    # 'Alle kouluikäisten lasten taloudet, year (TE)',
    # 'te_aklap',
    # 'Kouluikäisten lasten taloudet, year (TE)',
    # 'te_klap',
    # 'Teini-ikäisten lasten taloudet, year (TE)',
    # 'te_teini',
    # 'Aikuisten taloudet, year (TE)',
    'te_aik',
    # 'Eläkeläisten taloudet, year (TE)',
    'te_elak',
    # 'Omistusasunnoissa asuvat taloudet, year (TE)',
    'te_omis_as',
    # 'Vuokra-asunnoissa asuvat taloudet, year (TE)',
    'te_vuok_as'
    # 'Muissa asunnoissa asuvat taloudet, year (TE)'
    # 'te_muu_as'
]


def get_nominal_cols(year):
    return [re.sub('year', year, s) for s in NOMINAL_COLUMNS]


def round_pono(pono, level):
    div = 10 ** (level - 4)
    return pono - pono % div


def impute_with_class_mean(data, column_to_impute, based_on='rakennukset_bin'):
    if column_to_impute not in ['pono', 'pono.level', 'vuosi', 'nimi']:
        df = data.drop(labels=['pono', 'pono.level', 'vuosi', 'nimi'], axis=1)
        val_table = df.groupby(by=based_on)[column_to_impute].describe()['mean']
        data_predict = df.loc[df[column_to_impute].isnull(), :]

        return [val_table[row.rakennukset_bin] for index, row in data_predict.iterrows()]
    return []


def impute_with_regression(data, column_to_impute):
    df = data.drop(labels=['pono', 'pono.level', 'vuosi', 'nimi'], axis=1)
    # select only cols with all values
    cols = df.dropna(axis=1, inplace=False).columns.values.tolist()
    # df.dropna(axis=1, inplace=True)
    df_imp = df.dropna(axis=0).loc[:, cols + [column_to_impute]]
    lr = LinearRegression().fit(X=df_imp.drop(labels=column_to_impute, axis=1), y=df_imp.loc[:, column_to_impute])
    data_predict = df.loc[df[column_to_impute].isnull(), cols]

    return lr.predict(data_predict)


def check_class_means(data):
    for col in data.columns.values:
        if col not in ['pono', 'pono.level', 'vuosi', 'nimi']:
            print(col)
            print(data.groupby(by='rakennukset_bin')[col].describe()['mean'])


def merge_and_clean_data(data, data2):
    # combine nominal and share vars
    data = data.loc[:,
           list(chain.from_iterable([['pono', 'pono.level', 'vuosi', 'nimi'], NOMINAL_VARS]))]
    data2 = data2.loc[:, list(chain.from_iterable([['pono', 'vuosi', 'nimi'], SHARES_VARS]))]
    cols_to_use = data2.columns.difference(data.columns)
    data = pd.merge(data, data2[cols_to_use], left_index=True, right_index=True, how='outer')
    data = data.reindex()
    data.loc[data['pono.level'] == 5, 'pono'] = [format(x, '05d') for x in data.loc[data['pono.level'] == 5, 'pono']]
    data.loc[data['pono.level'] == 3, 'pono'] = [format(x * 100, '05d') for x in
                                                 data.loc[data['pono.level'] == 3, 'pono']]
    data.loc[data['pono.level'] == 2, 'pono'] = [format(x * 100, '05d') for x in
                                                 data.loc[data['pono.level'] == 2, 'pono']]
    to_format = ['he_kika', 'tr_mtu', 'ra_as_kpa', 'hr_pi_tul', 'hr_ke_tul', 'hr_hy_tul',
                 'pt_tyoll', 'pt_tyott', 'pt_tyovu', 'pt_0_14', 'pt_opisk', 'pt_elakel', 'hr_ovy',
                 'tr_pi_tul', 'tr_ke_tul', 'tr_hy_tul', 'te_nuor', 'te_eil_np', 'te_laps', 'te_aik',
                 'te_elak', 'te_omis_as', 'te_vuok_as', 'te_takk', 'te_as_valj']
    for c in to_format:
        data[c] = [float(str(x).replace(",", ".")) for x in data[c]]
    cut_to_bins(data, 'ra_asrak', 'rakennukset_bin')
    return data


def cut_to_bins(data, col_base, new_col_name):
    data[new_col_name] = pd.cut(data[col_base], 5, retbins=False, labels=False)
    return data


def get_edu_data():
    edu = pd.read_csv('education.csv', sep=";", dtype={'Yliopistot': 'O', 'AMK': 'O'}, usecols=['Yliopistot', 'AMK'])
    amk = edu.AMK.value_counts()
    yliop = edu.Yliopistot.value_counts()
    return amk, yliop