from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn import preprocessing
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import locale


warnings.filterwarnings('ignore')
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')


def pre_processing(data):

    encoding = preprocessing.LabelEncoder()

    encoding.fit(data['Dia da Semana'])
    data['Dia da Semana'] = encoding.transform(data['Dia da Semana'].copy())

    return data


def print_result(predictions):

    begin = '2023-01-21'
    end = '2023-01-25'
    period = pd.date_range(begin, end).tolist()
    result = pd.DataFrame(period, columns=['Data'])

    result['Vendas'] = np.around(predictions[:5]).astype(int)
    print(result)
    print(f"Total:\t      {sum(result['Vendas'])}")


def linear_regression(data):

    print('\nRegressão Linear\n')
    size = len(data)

    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(np.arange(size).reshape(-1, 1), data)

    predictions = lr_model.predict(np.arange(size + 1, size + 6).reshape(-1, 1))

    final = np.around(predictions).astype(int)
    print_result(final)
    print(f'\nRMSE: {np.sqrt(mean_squared_error(np.arange(size).reshape(-1, 1), data))}')


def xgboost_regression(data):

    print('\n\tXGBoost\n')
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

    begin = '2023-01-21'
    end = '2023-01-27'
    period = pd.date_range(begin, end).tolist()

    new_data = pd.DataFrame(period, columns=['Data'])
    new_data['Dia da Semana'] = new_data['Data'].dt.day_name('pt_BR.UTF-8')
    new_data = new_data.drop(['Data'], axis='columns')
    new_data = pre_processing(new_data.copy())

    multi_xgb_model = MultiOutputRegressor(xgb_regressor, n_jobs=-1).fit(train, test)
    predictions = multi_xgb_model.predict(new_data)

    final = np.around(predictions[:5]).astype(int)
    print_result(final)
    print(f'\nRMSE: {np.sqrt(mean_squared_error(train, test))}')


def arima_based_regressors(data):

    print('\n\tARIMA\n')
    train = data.drop(['Data'], axis='columns')

    arima_regressor = ARIMA(train, order=(1, 2, 1))
    arima_model = arima_regressor.fit()
    arima_predictions = arima_model.forecast(steps=5)

    print_result(arima_predictions.tolist())
    print(f'\nRMSE: {np.sqrt(pow(arima_model.resid, 2).mean())}')

    print('\n\tSARIMAX\n')
    sarimax_regressor = SARIMAX(train, order=(1, 1, 0), seasonal_order=(1, 1, 0, 7))
    sarimax_model = sarimax_regressor.fit(disp=False)
    sarimax_predictions = sarimax_model.forecast(steps=5)

    print_result(sarimax_predictions.tolist())
    print(f'\nRMSE: {np.sqrt(pow(sarimax_model.resid, 2).mean())}')

    # TODO: Analisar ferramenta para selecionar de forma automática qual seria o modelo mais adequado para este caso
    # TODO: Refinar hiperparâmetros dos regressores ARIMA
