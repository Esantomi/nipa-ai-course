from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from elice_utils import EliceUtils
from tensorflow.keras.layers import Dense, LSTM, ReLU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, ReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE
elice_utils = EliceUtils()

def rev_min_max_func(scaled_val):

    df = pd.read_csv('data/stock.csv') 
    df = df.fillna(0)
    max_val = max(df['Close'])
    min_val = min(df['Close'])
    og_val = (scaled_val * (max_val - min_val)) + min_val
    return og_val

def data_plot():

    # --- 주식 데이터 로드, 전처리, 분할, 모델 학습하기(이전 실습에서 진행) --- #
    df = pd.read_csv('data/stock.csv') 

    # 데이터프레임 출력(데이터프레임은 (헹 X 열)로 이루어진 표 형태의 특수한 데이터 타입)
    print(df)


    # --- 주식 데이터 살펴보기 --- #

    print('\n주식 데이터의 형태를 출력')
    print(df.shape)

    print('\n주식 데이터의 정보를 출력')
    print(df.info)

    print('\n주식 데이터의 상단 5개 행을 출력')
    print(df.head())

    print('\n주식 데이터의 하단 5개 행을 출력')
    print(df.tail())

    print('\n주식 데이터의 모든 열을 출력')
    print(df.columns)

    print('\n주식 데이터의 요약 통계 자료 출력')
    print(df.describe())

def train():

    # Keras 딥러닝 모델 학습을 위한 파라미터(옵션값)을 설정합니다.
    learning_rate = 0.01
    epochs = 1000
    batch_size = 200
    input_size = 5
    time_step = 1 

    # --- 주식 데이터 로드, 전처리, 분할, 모델 학습하기(이전 실습에서 진행) --- #
    df = pd.read_csv('data/stock.csv') 

    # 주가의 중간값 계산하기
    high_prices = df['High'].values
    low_prices = df['Low'].values
    mid_prices = (high_prices + low_prices) / 2

    # 주가 데이터에 중간 값 요소 추가하기
    df['Mid'] = mid_prices

    # 종가의 5일 이동평균값을 계산하고 주가 데이터에 추가하기
    ma5 = df['Adj Close'].rolling(window=5).mean()
    df['MA5'] = ma5

    df = df.fillna(0) # 비어있는 값을 모두 0으로 바꾸기

    # Date 열를 제거합니다.
    df = df.drop('Date', axis = 1)

    # 데이터 스케일링(MinMaxScaler 적용)
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(df)

    output = min_max_scaler.transform(df)
    output = pd.DataFrame(output, columns=df.columns, index=list(df.index.values))

    x_data = output[['Open', 'Low', 'High', 'Mid', 'MA5']]
    y_data = output['Close']

    train_size = int(len(x_data) * 0.7)
    test_size = int(len(y_data) * 0.3) + train_size

    train_x = np.array(x_data[:train_size])
    train_x = train_x.reshape((train_x.shape[0], time_step, input_size))
    train_y = np.array(y_data[:train_size])

    """
    validation_x = np.array(x_data[train_size:validation_size])
    validation_x = validation_x.reshape((validation_x.shape[0], time_step, input_size))
    validation_y = np.array(y_data[train_size:validation_size])
    """
    test_x = np.array(x_data[train_size:])
    test_x = test_x.reshape((test_x.shape[0], time_step, input_size))
    test_y = np.array(y_data[train_size:])

    # Keras 모델을 생성합니다.
    model = Sequential()


    # 생성된 딥러닝 모델에 학습용 데이터(train_x)를 넣습니다.
    model.add(LSTM(512, input_shape=(time_step, input_size)))
    model.add(Dense(1))
    model.add(ReLU())
    model.compile(optimizer=RMSprop(), loss=MSE, metrics=['mae', 'mape'])

    # 데이터를 학습을 진행합니다.
    model.summary()
    history = model.fit(train_x, train_y, epochs=epochs,   
                        batch_size=batch_size, verbose=1)
    model.evaluate(test_x, test_y, verbose=0)
    

    # --- 학습 결과를 그래프로 확인해 봅니다 --- #

    # 학습된 모델로부터 테스트 데이터를 예측합니다.
    pred = model.predict(test_x)

    fig = plt.figure(facecolor='white', figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(rev_min_max_func(test_y), label='True') # 실제 주가
    ax.plot(rev_min_max_func(pred), label='Prediction') # 우리가 만든 딥러닝 모델이 예측한 주가
    ax.legend()

    # 현재까지 그려진 그래프를 시각화
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")