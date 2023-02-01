from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_dist(x, y, title, xlabel, ylabel):

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    major_ticks = np.arange(0, 2001, 100)
    minor_ticks = np.arange(0, 47, 5)

    ax.set_xticks(minor_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_yticks(major_ticks)
    ax.set_yticks(major_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)

    plt.show()


def time_series_build(predictor):

    x = np.array(predictor)
    my_ts = TimeSeriesSplit(n_splits=8, test_size=5)

    print('TIME SERIES:\n')
    for (train, test) in my_ts.split(x):
        print(f'Training indexes = {train}')
        print(f'Testing indexes = {test}\n')


def main():

    os.chdir('..')
    frexco_dataset = pd.read_excel(os.getcwd() + '/data/Dados.xlsx', usecols=['Data', 'Vendas'])

    # print(frexco_dataset.head())

    sales = frexco_dataset.drop(['Data'], axis='columns')
    dates = frexco_dataset['Data'].apply(lambda x: x.strftime('%d-%m-%y'))

    # print(predictor.head())
    # print(dates.head())

    time_series_build(sales['Vendas'].tolist())
    plot_dist(dates, sales, 'Demanda de Alimentos por Dia (Frexco)', 'Datas', 'Demanda')


if __name__ == '__main__':
    main()
