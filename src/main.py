from ml_models import arima_based_regressors
from data_visualization import decomposition_plot
from data_visualization import correlation_plots
from data_visualization import plot_time_series
from data_visualization import plot_week_series
from ml_models import xgboost_regression
from ml_models import linear_regression
from data_visualization import week_boxplot
from data_visualization import diff_plot
import pandas as pd
import os


def main():

    os.chdir('..')
    frexco_dataset = pd.read_excel(os.getcwd() + '/data/Dados.xlsx', usecols=['Data', 'Vendas'])

    print(frexco_dataset.head())
    print(frexco_dataset.tail())
    print(frexco_dataset.isna().sum())

    sales = frexco_dataset.drop('Data', axis='columns')
    dates = frexco_dataset['Data'].apply(lambda x: x.strftime('%d-%m-%Y'))

    print(sales.head())
    print(dates.head())

    # ANÁLISE DOS DADOS
    print(sales.describe().transpose())
    plot_time_series(dates, sales, 'Demanda Diária de Alimentos (Frexco)', 'Datas', 'Demanda')
    plot_week_series(frexco_dataset.copy())
    week_boxplot(frexco_dataset.copy())
    decomposition_plot(frexco_dataset.copy())
    diff = diff_plot(sales.copy())
    correlation_plots(diff.copy())

    # MODELOS
    linear_regression(sales.copy())
    xgboost_regression(frexco_dataset.copy())
    arima_based_regressors(frexco_dataset.copy())

    # TODO: Comparar o aprendizado dos modelos em gráfico, assim como as métricas de erro (RMSE)
    # TODO: Refatorar código (limpeza e otimização)


if __name__ == '__main__':
    main()
