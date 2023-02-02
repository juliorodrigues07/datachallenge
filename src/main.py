from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os


warnings.filterwarnings('ignore')
color_pal = sns.color_palette()
# plt.style.use('fivethirtyeight')


def plot_time_series(x, y, title, xlabel, ylabel):

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(x, y)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    major_ticks = np.arange(0, 1801, 100)
    minor_ticks = np.arange(0, 47, 5)

    ax.set_xticks(minor_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_yticks(major_ticks)
    ax.set_yticks(major_ticks, minor=True)

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)

    plt.show()


def plot_week_series(dataset):

    dataset['Data'] = pd.to_datetime(dataset['Data'], infer_datetime_format=True)
    dataset = dataset.set_index('Data')

    week = dataset.loc[(dataset.index >= '2022-12-12') & (dataset.index <= '2022-12-16')]
    weekend = dataset.loc[(dataset.index >= '2022-12-16') & (dataset.index <= '2022-12-18')]

    fig, ax = plt.subplots(figsize=(15, 12))

    week.plot(ax=ax, label='Segunda a Sexta', title='Semana de Vendas', color='green')
    weekend.plot(ax=ax, label='Fim de Semana', color='red')

    ax.axvline('2022-12-16', color='black', ls='--')
    ax.legend(['Segunda a Sexta', 'Fim de Semana'])

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)
    plt.show()


def main():

    os.chdir('..')
    frexco_dataset = pd.read_excel(os.getcwd() + '/data/Dados.xlsx', usecols=['Data', 'Vendas'])

    # print(frexco_dataset.head())
    # print(frexco_dataset.tail())
    # print(frexco_dataset.isna().sum())

    sales = frexco_dataset.drop('Data', axis='columns')
    dates = frexco_dataset['Data'].apply(lambda x: x.strftime('%d-%m-%Y'))

    # print(sales.head())
    # print(dates.head())

    # Sumário
    print(sales.describe().transpose())
    plot_time_series(dates, sales, 'Demanda Diária de Alimentos (Frexco)', 'Datas', 'Demanda')
    plot_week_series(frexco_dataset)


if __name__ == '__main__':
    main()
