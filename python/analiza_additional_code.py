from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import cm
import pandas as pd
import numpy as np
import os

from analiza_danych.analiza_geolocation import geolocate
from analiza_danych.analiza_mapping import plot_points_geolocation

from analiza_danych.analiza_mysql_db_serwer import table_management


#dane do logowania do bazy - stock_market_private
hostname_priv = "51.38.128.241"
dbname_priv = "stock_market_private"
uname_priv = "remoteusr"
pwd_priv = "b2uMChfEUnynGzh"



def check_long_texts(string_lst):
    """ funkcja do wstawiania spacji jeśli są długie teksty """
    subs_lst = []
    for company in string_lst:
        string_middle = len(company) / 2

        space_to_replace = None
        for i in range(len(company)):
            if i > string_middle and company[i] == " ":
                space_to_replace = i
                break
        if space_to_replace is not None:
            company = company[:space_to_replace] + "\n" + company[space_to_replace + 1:]
        company_str = company.strip()
        subs_lst.append(company_str)
    return subs_lst



def read_data_from_excel(file_name):
    return pd.read_excel(f"C:\\Users\\m_met\OneDrive\\Pulpit\\tuinwestor analizy danych\\fotowoltaika\\{file_name}")


def geolocate_city(city, country):
    """ funkcja do geolokalizowania miejscowości - zwracane są współrzędne (lat, lng) """
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(city + ',' + country)

    return {
        'latitude': location.latitude,
        'longitude': location.longitude,
        'address': location.address}


def save_csv_file(data, file_name):
    """ zapisywanie pliku vsc we wskazanej lokalizacji """
    if data is not None and data.empty is False:
        directory = "C:\\Users\\m_met\\OneDrive\\Pulpit\\tuinwestor analizy danych"
        full_path = (os.path.join(directory, file_name))
        data.to_excel(full_path, index=True)



class transform_data:
    """ klasa z funkcjami do przetwarzania i edytowania danych """

    def cumulative_sum(self, data):
        """ liczenie skumulowanego wzrostu dla danych - po miesiącach """
        data = data[['data_wplywu_wniosku']].copy()
        data = data[(data['data_wplywu_wniosku'].dt.year > 2018) & (data['data_wplywu_wniosku'].dt.year < 2022)]

        data['counter'] = 1

        data = data.groupby(pd.Grouper(key='data_wplywu_wniosku', axis=0, freq='Y')).sum()
        data = data.cumsum()
        return data

    def cum_pv_projects(self, df):
        """ wizualizacja skumulowanej ilości farm fotowoltaicznych """
        #df = df[['data_wplywu_wniosku']]
        df['counter'] = 1

        print(df.columns)

        # df = df.groupby(pd.Grouper(key='data_wplywu_wniosku', axis=0, freq='M')).sum()
        # df = df.cumsum()
        # plot_one_line(data=df['counter'], chart_title="Skumulowana liczba farm wiatrowych")

        df = df[['state', 'data_wplywu_wniosku']].copy()
        df['woj_counter'] = 1
        df['date_counter'] = 1

        df = df.set_index('state')
        df = df[df['data_wplywu_wniosku'].dt.year < 2022].sort_values(by='data_wplywu_wniosku')

        df = df.groupby([df.index, pd.Grouper(key='data_wplywu_wniosku', axis=0, freq='M')])['date_counter'].sum()
        df = df.groupby(level=0).cumsum()
        # df['woj'] = df.index.get_level_values(0)
        # df['month'] = df.index.get_level_values(1)

        df = df.reset_index()
        print(df)
        df = df.pivot(index='data_wplywu_wniosku', columns='state', values='date_counter').reset_index()
        df = df.fillna(method="ffill").fillna(0)
        print(df)

        save_csv_file(data=df, file_name="fotowoltaika województwa.xlsx")

        #unique_woj = df.groupby(level=0).max().sort_values(ascending=False).index.tolist()  # dbierzemy listę według od największej ilości farm

        # plt.rc('figure', figsize=(15, 10))
        # plt.style.use('fivethirtyeight')
        # fig, ax = plt.subplots()
        #
        # # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 16))))
        # color_map = cm.get_cmap('copper', 16)
        # plt.gca().set_prop_cycle(cycler('color', color_map(np.linspace(1, 0, 16))))
        #
        # for woj in unique_woj:
        #     data = df[df.index.isin([woj], level=0)]
        #     # print(data)
        #     dates = data.index.get_level_values(1).unique().tolist()
        #     plt.plot(dates, data.values, label=woj, linewidth=2)  # plotting t, a separately
        #
        # ax.grid(False)
        # ax.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)
        # ax.yaxis.tick_right()
        #
        # plt.title("Skumulowany wzrost liczby działek - farmy fotowoltaiczne według województw", y=1.0, pad=-14,
        #           fontname="Times New Roman",
        #           fontweight="bold")
        # plt.legend()
        # plt.show()





def get_geo_loc(df, data):
    """ pobieranie geolokalizacji - draft """
    #data = get_data()
    df = data.dino_markets()
    # data.visualize_data(chart_title='Skumulowana liczba sklepów otwieranych przez Dino')
    data.save_csv_file(file_name="dino sklepy.xlsx")

    df = df.fillna(value=np.nan)

    frame = []

    for row in df.to_dict('records'):
        location = row['miasto']
        investor = row['nazwa_inwestor']

        print(location)
        if pd.isna(location) is False and location != '':
            geo_dictionary = geolocate_city(city=location, country='Poland')
            print(geo_dictionary)
            info = f'{location}, Poland \n{investor}'

            frame_dictionary = {
                'wpływ_wniosku': row['data_wplywu_wniosku'],
                'inwestor': investor,
                'info': info,
                'summary': row['nazwa_zam_budowlanego'],
                'hash_code': row['hash_code'],
            }

            frame_dictionary = {**frame_dictionary, **geo_dictionary}
            frame.append(frame_dictionary)
        # print(frame_dictionary)
        # break

    df = pd.DataFrame(frame)
    save_csv_file(data=df, file_name='Dino markets.xlsx')

    data.visualize_data(chart_title='Skumulowana liczba sklepów otwieranych przez Dino')


def dino_ebud_data(data):
    """ tymczasowa funkcja to zapisywania dnaych o Dino - data from get_data()"""
    import investpy
    df_stock = investpy.get_stock_historical_data(stock='DNP',
                                                  country='Poland',
                                                  from_date='01/01/2017',
                                                  to_date='15/03/2022')
    df_stock = df_stock[['Open']]
    df_raw = data.dino_markets()

    df = df_raw[['data_wplywu_wniosku']].copy()
    df['counter'] = 1

    df = df.groupby(pd.Grouper(key='data_wplywu_wniosku', axis=0, freq='D')).sum()
    df = df.rolling(min_periods=1, window=30).sum()
    # df = df.cumsum()
    df = df.asfreq('D')
    df = df.fillna(method='bfill')
    print(df)

    df_merge = pd.merge(df, df_stock, left_index=True, right_index=True)

    print(df_merge)

    save_csv_file(data=df_merge, file_name="dino rolling M.xlsx")



def dino_markets_by_year():
    """ sklepy Dino - rozwó r/r """
    year = 2021
    map_shp = "C:\\Users\\m_met\\OneDrive\\Pulpit\\tuinwestor analizy danych\Polska - województwa\\Województwa.shx"

    data = ''  # get_data_wnioski()
    data.dino_markets()
    df = data.prepare_for_mapping(color='red', label='Sieć sklepów')
    df = df.drop_duplicates(subset=['miasto', 'wojewodztwo']).copy()

    df['data_wplywu_wniosku'] = pd.to_datetime(df['data_wplywu_wniosku'])
    df = df[df['data_wplywu_wniosku'].dt.year <= year]
    print(len(df))

    df = df.rename(columns={'miasto': 'city', 'wojewodztwo': 'state'})
    geo_cities = geolocate(cities_df=df, city_col='city', filter_cols=['city', 'state'], country='Poland')
    df = geo_cities.core()

    title = f"Wzrost działalności Dino Polska SA (2016-{year})\n{len(df)} sklepów"

    mapping = plot_points_geolocation()
    mapping.core(df_geo=df, title=title, map_link=map_shp, size=30, differ_size=False, annotate=False)


def get_stock_price(ticker):
    """ pobieranie informacji o cenie """
    import json

    cls = table_management(hostname_priv, dbname_priv, uname_priv, pwd_priv)
    stock_data = cls.fetch_one_result_filtered('investing_com_trading_data', 'Close', f'ticker = "{ticker}"')
    cls.close_connection_2()

    stock_data_dict = json.loads(stock_data[0])
    frame = [[k, v] for k, v in stock_data_dict.items()]

    df = pd.DataFrame(frame, columns=['date', 'price'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    return df



