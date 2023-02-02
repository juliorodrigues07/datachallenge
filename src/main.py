from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
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


def linear_regression_test(data):

    size = len(data)
    begin = '2023-01-21'
    end = '2023-01-25'
    period = pd.date_range(begin, end).tolist()

    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(np.arange(size).reshape(-1, 1), data)

    predictions = lr_model.predict(np.arange(size + 1, size + 6).reshape(-1, 1))

    final = list()
    predictions = predictions.tolist()
    for x in predictions:
        final.append(x[0])

    predictions = pd.Series(final, index=period)
    print(predictions)


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
    # print(sales.describe().transpose())
    # plot_time_series(dates, sales, 'Demanda Diária de Alimentos (Frexco)', 'Datas', 'Demanda')
    # plot_week_series(frexco_dataset)
    # week_boxplot(frexco_dataset)

    frexco_dataset['Dia da Semana'] = frexco_dataset['Data'].dt.day_name('pt_BR.UTF-8')

    # MODELOS
    # linear_regression_test(sales)
    xgb_regressor = xgb.XGBRegressor(base_score=0.5,
                                     booster='gbtree',
                                     n_estimators=50,
                                     early_stopping_rounds=10,
                                     objective='reg:linear',
                                     max_depth=3,
                                     learning_rate=0.03)

    train = frexco_dataset.drop(['Data'], axis='columns')
    test = frexco_dataset['Vendas']

    encoding = preprocessing.LabelEncoder()
    encoding.fit(train['Dia da Semana'])
    train['Dia da Semana'] = encoding.transform(train['Dia da Semana'].copy())

    xgb_regressor.fit(train, test, eval_set=[(train, test), (train, test)], verbose=100)

    begin = '2023-01-21'
    end = '2023-01-25'
    period = pd.date_range(begin, end).tolist()

    new_data = pd.DataFrame(period, columns=['Data'])
    new_data['Dia da Semana'] = new_data['Data'].dt.day_name('pt_BR.UTF-8')
    new_data = new_data.drop(['Data'], axis='columns')

    encoding.fit(new_data['Dia da Semana'])
    new_data['Dia da Semana'] = encoding.transform(new_data['Dia da Semana'].copy())

    # predictions = xgb_regressor.predict(new_data)
    # print(predictions)


if __name__ == '__main__':
    main()
