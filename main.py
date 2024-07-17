import pandas as pd
import streamlit as st
from merlion.models.forecast.prophet import ProphetConfig, Prophet
from merlion.utils import TimeSeries
from merlion.models.forecast.trees import LGBMForecaster, LGBMForecasterConfig
from merlion.models.forecast.arima import Sarima, SarimaConfig
from merlion.evaluate.forecast import ForecastMetric
import matplotlib.pyplot as plt
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

st.title('Считаем временные ряды')
st.caption('вместе с Серёжей')
st.divider()

# Инициализация переменных состояния сессии
if 'submit_m' not in st.session_state:
    st.session_state.submit_m = None

if 'file' not in st.session_state:
    st.session_state.file = None

if 'conf' not in st.session_state:
    st.session_state.conf = ""

st.write('Загрузи датафрейм, в котором первый столбец - даты, а второй - таргет')

with st.sidebar:
    st.write('Панель управления')
    upload = st.file_uploader(label='Грузи свой временной ряд, бро', type='csv')
    if upload:
        st.session_state.file = pd.read_csv(upload, index_col=0, parse_dates=True)
    button = st.button('Использовать тестовый файл')
    if button:
        st.session_state.file = pd.read_csv('test.csv', index_col=0, parse_dates=True)
    if st.session_state.file is not None and not st.session_state.file.empty:
        with st.form(key='model'):
            st.radio('Модель', options=['LGBM', 'Prophet', 'SARIMA'], key='model')
            submit_m = st.form_submit_button('Выбрать')
            if submit_m:
                st.session_state.submit_m = True

        if st.session_state.model == 'SARIMA':
            st.session_state.conf = st.text_input('Введи p d q P D Q S (через пробел)', st.session_state.conf)

if st.session_state.submit_m:
    if st.session_state.model == 'SARIMA' and not st.session_state.conf:
        st.warning('Введите конфигурацию для модели SARIMA и повторно нажмите "Выбрать"')
    else:
        train_size = int(len(st.session_state.file) * 0.8)
        data_train = st.session_state.file.iloc[:train_size]
        data_test = st.session_state.file.iloc[train_size:]

        train_series = TimeSeries.from_pd(data_train)
        test_series = TimeSeries.from_pd(data_test)

        if st.session_state.model == 'LGBM':
            config = LGBMForecasterConfig()
            model = LGBMForecaster(config)
            model.train(train_series)

        elif st.session_state.model == 'Prophet':
            config = ProphetConfig()
            model = Prophet(config)
            model.train(train_series)

        elif st.session_state.model == 'SARIMA':
            conf = st.session_state.conf.split()
            order_conf = [int(num) for num in conf[:3]]
            season_conf = [int(num) for num in conf[3:]]
            config = SarimaConfig(order=order_conf, seasonal_order=season_conf)
            model = Sarima(config)
            model.train(train_series)

        forecast, stderr = model.forecast(time_stamps=test_series.time_stamps)
        forecast_values = forecast.to_pd()

        mae = ForecastMetric.MAE.value(data_test.iloc[:, 0], forecast_values.iloc[:, 0])

        fig = plt.figure(figsize=(20, 15))
        plt.plot(data_train.index, data_train.iloc[:, 0], label='Train')
        plt.plot(data_test.index, data_test.iloc[:, 0], label='Test', color='orange')
        plt.plot(forecast_values.index, forecast_values.iloc[:, 0], label='Forecast', color='red')
        plt.legend()
        plt.title('Model Forecast vs Actuals')

        st.pyplot(fig)
        st.write(f'MAE: {mae:.2f}')
