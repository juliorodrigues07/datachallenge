from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import locale


color_pal = sns.color_palette()
# plt.style.use('fivethirtyeight')
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')


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
    ax.legend(['Vendas'], loc='upper right')

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
    ax.legend(['Segunda a Sexta', 'Fim de Semana'], loc='best')

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)
    plt.show()


def week_boxplot(dataset):

    dataset['Dia da Semana'] = dataset['Data'].dt.day_name('pt_BR.UTF-8')

    frexco_dataset = dataset.set_index('Data')
    frexco_dataset.index = pd.to_datetime(frexco_dataset.index)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=frexco_dataset, x='Dia da Semana', y='Vendas', palette='Blues')

    ax.axvline(x=3.5, color='red', ls='--')
    ax.axvline(x=5.5, color='red', ls='--')
    ax.set_title('Demanda por Dia da Semana')

    plt.show()


def decomposition_plot(dataset):

    multiplicative_decomposition = seasonal_decompose(dataset['Vendas'], model='multiplicative', period=7)

    plt.rcParams.update({'figure.figsize': (16, 12)})
    multiplicative_decomposition.plot().suptitle('Decomposição da Série Temporal', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def diff_plot(data):

    first_diff = data.diff()
    stationary_test(data.dropna())
    second_diff = first_diff.diff()
    stationary_test(second_diff.dropna())

    fig, ax = plt.subplots(figsize=(10, 8))

    data.plot(ax=ax, label='Original', color='blue')
    first_diff.plot(ax=ax, label='Primeira Ordem', color='green')
    second_diff.plot(ax=ax, label='Segunda Ordem', color='red')

    ax.set_title('Diferenciação nas Séries Temporais', fontsize=14)
    ax.set_ylabel("Vendas", fontsize=12)
    ax.set_xlabel("Dias", fontsize=12)

    ax.grid(which='minor')
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)

    plt.legend(['Original', 'Primeira Ordem', 'Segunda Ordem'], loc='lower left')
    plt.show()

    return second_diff.dropna()


def correlation_plots(diff):

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(diff, title='Autocorrelação', lags=16, ax=ax1, color='mediumblue')
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(diff, title='Autocorrelação Parcial', lags=16, ax=ax2, color='mediumblue')

    plt.tight_layout()
    plt.show()


def stationary_test(data):

    result = adfuller(data)
    p_value = result[1]
    n_lags = result[2]
    print(f'p-value: {p_value}, lags: {n_lags}')

    if p_value <= 0.05:
        print('Rejeita H0, dados estão estacionários!')
    else:
        print("Não rejeita H0 (hipótese fraca), o que indica que os dados não são estacionários")
