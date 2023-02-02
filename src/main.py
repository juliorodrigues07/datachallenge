from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import locale
import os


warnings.filterwarnings('ignore')
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
    multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
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
    fig = sm.graphics.tsa.plot_acf(diff, title='Autocorrelação', lags=16, ax=ax1, color='mediumblue')
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff, title='Autocorrelação Parcial', lags=16, ax=ax2, color='mediumblue')

    plt.tight_layout()
    plt.show()


def pre_processing(data):

    encoding = preprocessing.LabelEncoder()

    # Dicretiza atributos categóricos ('ruim', 'regular', 'bom', ...) -> (0, 1, 2, ...)
    encoding.fit(data['Dia da Semana'])
    data['Dia da Semana'] = encoding.transform(data['Dia da Semana'].copy())

    return data


def stationary_test(data):

    result = adfuller(data)
    p_value = result[1]
    n_lags = result[2]
    print(f'p-value: {p_value}, lags: {n_lags}')

    if p_value <= 0.05:
        print('Rejeita H0, dados estão estacionários!')
    else:
        print("Não rejeita H0 (hipótese fraca), o que indica que os dados não são estacionários")


def linear_regression_test(data):

    print('\nRegressão Linear\n')
    size = len(data)
    begin = '2023-01-21'
    end = '2023-01-25'
    period = pd.date_range(begin, end).tolist()

    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(np.arange(size).reshape(-1, 1), data)

    predictions = lr_model.predict(np.arange(size + 1, size + 6).reshape(-1, 1))

    result = pd.DataFrame(period, columns=['Data'])
    result = result.set_index('Data')

    result['Vendas'] = np.around(predictions).astype(int)
    print(result)


def xgboost_regression(data):

    print('\nXGBoost\n')
    test = data.drop(['Data'], axis='columns')

    data['Dia da Semana'] = data['Data'].dt.day_name('pt_BR.UTF-8')

    train = data.drop(['Data', 'Vendas'], axis='columns')
    train = pre_processing(train.copy())

    xgb_regressor = xgb.XGBRegressor(base_score=0.5,
                                     booster='gbtree',
                                     n_estimators=250,
                                     n_jobs=-1,
                                     objective='reg:squarederror',
                                     max_depth=5,
                                     learning_rate=0.03)

    # Período para previsão (Deve conter os 7 dias para a codificação bater)
    begin = '2023-01-21'
    end = '2023-01-27'
    period = pd.date_range(begin, end).tolist()

    # Cria uma base de dados contendo as datas para previsão
    new_data = pd.DataFrame(period, columns=['Data'])
    new_data['Dia da Semana'] = new_data['Data'].dt.day_name('pt_BR.UTF-8')
    new_data = new_data.drop(['Data'], axis='columns')
    new_data = pre_processing(new_data.copy())

    multi_xgb_model = MultiOutputRegressor(xgb_regressor).fit(train, test)
    predictions = multi_xgb_model.predict(new_data)

    end = '2023-01-25'
    period = pd.date_range(begin, end).tolist()
    result = pd.DataFrame(period, columns=['Data'])
    result = result.set_index('Data')

    result['Vendas'] = np.around(predictions[:5]).astype(int)
    print(result)


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

    # ANÁLISE DOS DADOS
    print(sales.describe().transpose())
    plot_time_series(dates, sales, 'Demanda Diária de Alimentos (Frexco)', 'Datas', 'Demanda')
    plot_week_series(frexco_dataset.copy())
    week_boxplot(frexco_dataset.copy())
    diff = diff_plot(sales.copy())
    correlation_plots(diff.copy())

    # MODELOS
    linear_regression_test(sales)
    xgboost_regression(frexco_dataset)


if __name__ == '__main__':
    main()
