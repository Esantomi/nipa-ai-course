import numpy as np 
import pandas as pd 


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# to avoid warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, average_precision_score

from elice_utils import EliceUtils
elice_utils = EliceUtils()

def preprocess():

    #print("데이터 읽는 중...")
    data = pd.read_csv('data/uci-secom.csv')
    
    #print(data.isnull().sum())
    #print("결측 데이터 값을 처리합니다.")
    data = data.replace(np.NaN, 0)
    #print(data.isnull().sum())
    
    # 쓸모없는 데이터 지우기
    data = data.drop(columns = ['Time'], axis = 1)
    
    # 타겟 데이터 분리
    x = data.iloc[:,:590]
    y = data.iloc[:, 590]

    #print("학습용 데이터와 테스트용 데이터로 분리 합니다.")

    # Under sampling 수행
    failed_tests = np.array(data[data['Pass/Fail'] == 1].index)
    no_failed_tests = len(failed_tests)

    normal_indices = data[data['Pass/Fail'] == -1]
    no_normal_indices = len(normal_indices)
    
    np.random.seed(10)
    random_normal_indices = np.random.choice(no_normal_indices, size = no_failed_tests, replace = True)
    random_normal_indices = np.array(random_normal_indices)

    under_sample = np.concatenate([failed_tests, random_normal_indices])
    undersample_data = data.iloc[under_sample, :]

    x = undersample_data.iloc[:, undersample_data.columns != 'Pass/Fail'] 
    y = undersample_data.iloc[:, undersample_data.columns == 'Pass/Fail']
    y = np.ravel(y)

    x_train_us, x_test_us, y_train_us, y_test_us = train_test_split(x, y, test_size = 0.2, random_state = 4)
    
    #print("학습용 데이터 크기: {}".format(x_train_us.shape))
    #print("테스트용 데이터 크기: {}".format(x_test_us.shape))
    
    return x_train_us, x_test_us, y_train_us, y_test_us
    
    
def train(x_train_us, y_train_us):

    """
    #print("학습을 수행합니다.")
    
    model = XGBClassifier(random_state=2)
    
    # 파라미터 튜닝
    parameters = [{'max_depth' : [1, 2, 3, 4, 5, 6]}]

    grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'recall', cv = 4, n_jobs = -1)

    grid_search = grid_search.fit(x_train_us, y_train_us)
    
    # 베스트 모델 선택
    model = grid_search.best_estimator_
    """
    import pickle
    
    with open('model.pkl', 'rb') as f:
         model = pickle.load(f)
    
    return model
    
def evaluation():

    x_train_us, x_test_us, y_train_us, y_test_us = preprocess()
    model = train(x_train_us, y_train_us)
    # 테스트 데이터 예측
    y_pred = model.predict(x_test_us)
    print('평가 지표인 recall score를 출력합니다.: ', recall_score(y_test_us, y_pred), '\n')
    
    print("센서들의 중요도를 출력합니다.")
    xgb.plot_importance(model, height = 1, grid = True, importance_type = 'gain', show_values = False, max_num_features = 20)

    plt.rcParams['figure.figsize'] = (10, 15)
    plt.xlabel('The importance score for each features')
    plt.ylabel('Features')
    
    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
