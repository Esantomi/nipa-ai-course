# AI 실무 응용 과정 - 머신러닝 시작하기

실제 데이터와 코드를 다뤄보며 최신 머신러닝/딥러닝 프레임워크와 라이브러리를 배워요!
- [2021 NIPA AI 온라인 교육 - AI 실무 응용 과정](https://2021nipa.elice.io/tracks/1329/info)

## 목차

- [머신러닝 시작하기](#머신러닝-시작하기)
  * [01. 자료 형태의 이해](#01-자료-형태의-이해)
    + [자료 형태](#자료-형태)
    + [범주형 자료의 요약](#범주형-자료의-요약)
      - [도수 분포표 실습](#도수-분포표)
      - [막대 그래프 실습](#막대-그래프)
    + [수치형 자료의 요약](#수치형-자료의-요약)
      - [평균 실습](#평균)
      - [표준편차, 분산 실습](#표준편차-분산)
      - [히스토그램 실습](#히스토그램)
  * [02. 데이터 전 처리하기](#02-데이터-전-처리하기)
    + [데이터 전 처리의 역할](#데이터-전-처리의-역할)
    + [범주형 자료 전 처리](#범주형-자료-전-처리)
      - [대표적인 범주형 자료 변환 방식](#대표적인-범주형-자료-변환-방식)
      - [명목형 자료 변환하기 - 수치 맵핑 실습](#명목형-자료-변환하기---수치-맵핑)
      - [명목형 자료 변환하기 - 더미 방식 실습](#명목형-자료-변환하기---더미-방식)
    + [수치형 자료 전 처리](#수치형-자료-전-처리)
      - [대표적인 수치형 자료 변환 방식](#대표적인-수치형-자료-변환-방식)
      - [수치형 자료 변환하기 - 정규화 실습](#수치형-자료-변환하기---정규화)
      - [수치형 자료 변환하기 - 표준화 실습](#수치형-자료-변환하기---표준화)
    + [데이터 정제 및 분리하기](#데이터-정제-및-분리하기)
      - [결측값(Missing data) 처리하기](#결측값missing-data-처리하기) 
      - [이상치(Outlier) 처리하기](#이상치outlier-처리하기) 
      - [데이터 분리가 필요한 이유](#데이터-분리가-필요한-이유)
      - [지도학습 데이터 분리](#지도학습-데이터-분리)
      - [결측값 처리하기 실습](#결측값-처리하기)
      - [이상치 처리하기 실습](#이상치-처리하기)
      - [데이터 분리하기 실습](#데이터-분리하기)
  * [03. 지도학습 - 회귀](#03-지도학습---회귀)
    + [회귀 개념 알아보기](#회귀-개념-알아보기)
    + [단순 선형 회귀](#단순-선형-회귀)
      - [손실 함수(Loss function)](#단순-선형-회귀-모델의-손실-함수)
      - [경사 하강법(Gradient descent)](#단순-선형-회귀-모델의-경사-하강법) 
      - [단순 선형 회귀 분석하기 - 데이터 전 처리 실습](#단순-선형-회귀-분석하기---데이터-전-처리)
      - [단순 선형 회귀 분석하기 - 학습하기 실습](#단순-선형-회귀-분석하기---학습하기)
      - [단순 선형 회귀 분석하기 - 예측하기 실습](#단순-선형-회귀-분석하기---예측하기)
    + [다중 선형 회귀](#다중-선형-회귀)
      - [손실 함수(Loss fucntion)](#다중-선형-회귀-모델의-손실-함수) 
      - [경사 하강법(Gradient descent)](#다중-선형-회귀-모델의-경사-하강법) 
      - [다중 회귀 분석하기 - 데이터 전 처리 실습](#다중-회귀-분석하기---데이터-전-처리)
      - [다중 회귀 분석하기 - 학습하기 실습](#다중-회귀-분석하기---학습하기)
      - [다중 회귀 분석하기 - 예측하기 실습](#다중-회귀-분석하기---예측하기)
    + [회귀 평가 지표](#회귀-평가-지표)
      - [RSS - 단순 오차](#RSS---단순-오차) 
      - [MSE, MAE - 절대적인 크기에 의존한 지표](#MSE,-MAE---절대적인-크기에-의존한-지표)
      - [![image](https://user-images.githubusercontent.com/61646760/142835165-4f274d30-afec-40f0-bd7c-448be230194d.png) - 결정 계수](#R-squared---결정-계수)
      - [회귀 알고리즘 평가 지표 - MSE, MAE 실습](#회귀-알고리즘-평가-지표---MSE-MAE)
      - [회귀 알고리즘 평가 지표 - ![image](https://user-images.githubusercontent.com/61646760/142835165-4f274d30-afec-40f0-bd7c-448be230194d.png) 실습](#회귀-알고리즘-평가-지표---R2)
  * [04. 지도학습 - 분류](#04-지도학습---분류)
    + [분류 개념 알아보기](#분류-개념-알아보기)
    + [의사 결정 나무](#의사-결정-나무)
      - [의사 결정 나무 – 모델 구조](#의사-결정-나무---모델-구조) 
      - [간단한 의사 결정 나무 만들기 실습](#간단한-의사-결정-나무-만들기)
      - [의사 결정 나무 - 불순도](#의사-결정-나무---불순도)
      - [sklearn을 사용한 의사결정나무 - 데이터 전 처리 실습](#sklearn을-사용한-의사결정나무---데이터-전-처리)
      - [sklearn을 사용한 의사결정나무 - 학습하기 실습](#sklearn을-사용한-의사결정나무---학습하기)
      - [sklearn을 사용한 의사결정나무 - 예측하기 실습](#sklearn을-사용한-의사결정나무---예측하기)
    + [분류 평가 지표](#분류-평가-지표)
      - [혼동 행렬(Confusion Matrix)](#혼동-행렬)
      - [정확도(Accuracy)](#정확도Accuracy)
      - [정밀도(Precision)](#정밀도Precision)
      - [재현율(Recall,TPR)](#재현율Recall,-TPR)


# 머신러닝 시작하기

## 01. 자료 형태의 이해

### 자료 형태
- **수치형 자료(Numerical data)** : 양적 자료(Quantitative data) `예) 키, 몸무게, 성적, 나이`
  - **연속형 자료(Continuous data)** : 연속적인 관측값을 가짐 `예) 원주율(3.14159...), 시간`
  - **이산형 자료(Discrete data)** : 셀 수 있는 관측값을 가짐 `예) 뉴스 글자 수, 주문 상품 개수`
- **범주형 자료(Categorical data)** : 질적 자료(Qualitative data) `예) 성별, 지역, 혈액형`
  - **순위형 자료(Ordinal data)** : 범주 사이의 순서에 의미가 있음 `예) 학점 (A+, A, A-)`
  - **명목형 자료(Nominal data)** : 범주 사이의 순서에 의미가 없음 `예) 혈액형 (A, B, O, AB)`
- 혼동되는 경우
  - 범주형 자료는 숫자로 표기해도 범주형 자료 (남자 1, 여자 0)
  - 수치형 자료는 구간화하면 범주형 자료 (10 ~ 19세, 20 ~ 29세)

### 범주형 자료의 요약
- **도수 분포(Frequency distribution)**
  - 도수(Frequency) : 각 범주에 속하는 관측값의 개수
    - `value_counts()`
  - 상대 도수(Relative frequency) : 도수를 자료의 전체 개수로 나눈 비율
    - `(그 도수의 변량)/(총도수)`
    - `value_counts(normalize=True)`
  - 도수 분포표(Frequency table) : 범주형 자료에서 범주와 그 범주에 대응하는 도수, 상대 도수를 나열해 표로 만든 것
- **막대 그래프(Bar chart)** : 각 범주에서 도수의 크기를 막대로 그림
  - `plt.bar()`

#### 도수 분포표
```
import pandas as pd 
import numpy as np

# drink 데이터
drink = pd.read_csv("drink.csv")

# print(drink.head())

'''
   Attend Name
0       1    A
1       0    A
2       1    A
3       1    A
4       1    A
'''

# Attend : 참석한 경우 1, 참석하지 않은 경우 0
# Name : 참석자의 이름

"""
1. 도수 계산
"""
drink_freq = drink[drink['Attend']==1]['Name'].value_counts()  # 술자리 참석자(Attend == 1)를 이름으로 도수 구하기 

print("도수 분포표")
print(drink_freq)

'''
도수 분포표
A    4
B    3
D    2
C    2
E    1
Name: Name, dtype: int64
'''
```
- `drink[drink['Attend']==1]['Name']` 부분은 `drink[A][B]` 구조로 이해하면 된다. 이는 drink dataframe에서 A 조건, B 조건 모두에 부합하는 것이라는 뜻이다.

#### 막대 그래프
```
import matplotlib.pyplot as plt

# 술자리 참석 상대도수 데이터 
labels = ['A', 'B', 'C', 'D', 'E']
ratio = [4,3,2,2,1]
    
# 막대 그래프
fig, ax = plt.subplots()

"""
1. 막대 그래프를 만드는 코드를 작성해 주세요
"""
plt.bar(labels, ratio)

# 출력에 필요한 코드
plt.show()
```
![image](https://user-images.githubusercontent.com/61646760/139657061-97d270f2-e617-425a-a83a-79d3ec59844b.png)

### 수치형 자료의 요약
범주형 자료와 달리 수치로 구성되어 있기에 통계값을 사용한 요약이 가능하다.
- **평균(Mean)** : 관측값의 산술평균
  - `np.mean()`
- **분산(Variance)** : 각 관측값이 자료의 평균으로부터 떨어진 정도 (자료가 얼마나 흩어졌는지 숫자로 표현)
  ```
  from statistics import variance
  variance()
  ```
- **표준편차(Standard deviation, s)** : 분산의 양의 제곱근
  ```
  from statistics import stdev
  stdev()
  ```
- **히스토그램(Histogram)** : 수치형 자료를 일정한 범위를 갖는 범주로 나누고 막대 그래프와 같은 방식으로 그림
  - `plt.hist()`
  - x축은 계급, y축은 빈도(도수, 상대 도수)
  - 도수를 비교할 때, 범주형 자료는 막대 그래프, 수치형 자료는 히스토그램을 쓴다.

#### 평균
```
import numpy as np

# 카페별 카페인 함량 정보
coffee = np.array([202,177,121,148,89,121,137,158])

"""
1. 평균 계산
"""
cf_mean = coffee.mean()
# cf_mean = np.mean(coffee)        # 위와 동일

# 소수점 둘째 자리까지 반올림하여 출력합니다. 
print("Mean :", round(cf_mean,2))  # Mean : 144.12
```

#### 표준편차, 분산
```
from statistics import stdev, variance
import numpy as np

# 카페별 카페인 함량 정보
coffee = np.array([202,177,121,148,89,121,137,158])

"""
1. 표준편차 계산
"""
cf_std = stdev(coffee)

# 소수점 둘째 자리까지 반올림하여 출력합니다. 
print("Sample std.Dev : ", round(cf_std,2))   # Sample std.Dev :  35.44

"""
2. 분산 계산
"""
cf_var = variance(coffee)

print("Sample variance : ", round(cf_var,2))  # Sample variance :  1256
```

#### 히스토그램
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 카페인 함량 데이터
coffee = np.array([202,177,121,148,89,121,137,158])

fig, ax = plt.subplots()

"""
1. 히스토그램을 그리는 코드를 작성해 주세요
"""
plt.hist(coffee)

# 히스토그램을 출력합니다.
plt.show()
fig.savefig("hist_plot.png")
```
![image](https://user-images.githubusercontent.com/61646760/139792945-0f5c3b2e-0f4c-4520-a5cc-978f720c0ba9.png)


## 02. 데이터 전 처리하기

### 데이터 전 처리의 역할
1. 머신러닝의 입력 형태로 **데이터 변환 (특성 엔지니어링)**
    - 대부분의 머신러닝 모델은 숫자 데이터를, 일반적으로 **행렬 형태**로 입력받는다.
    - 그러므로 전 처리를 통해 일반 데이터를 머신러닝 모델이 이해할 수 있는 **수치형 자료**로 변환해 주어야 한다.
2. 결측값 및 이상치를 처리하여 **데이터 정제**
    - NaN과 outlier 제거
3. 학습용 및 평가용 **데이터 분리**
    - 원본 데이터를 학습용 데이터와 평가용 데이터로 분리
    - 원본 데이터 전체를 학습용 데이터로 사용할 경우, 과적합 문제가 발생할 수 있다.

### 범주형 자료 전 처리
#### 대표적인 범주형 자료 변환 방식
- 명목형 자료
  - **수치 맵핑 방식**
    - 범주가 2개인 경우, 범주를 0, 1로 맵핑 `예) male 0, female 1`
    - 범주가 3개 이상인 경우, 수치의 크기 간격을 같게 하여 수치 맵핑 `예) S는 0, Q는 1, C는 2 (Embarked)`
  - **더미(Dummy) 기법**
    - 가변수(Dummy variable)를 사용해 각 범주를 0 또는 1로 변환
    - `예) Sex를 Sex_female과 Sex_male의 범주로 나누면, 남자는 Sex_female이 0, Sex_male이 1`
    - `Embarked를 Embarked_S, Embarked_Q, Embarked_C로 나누면, Q는 Embarked_S가 0, Embarked_Q가 1, Embarked_C가 0`
- 순서형 자료
  - 수치 맵핑 방식
    - 수치에 맵핑하여 변환하지만, 수치 간 크기 차이는 커스텀 가능 `예) 없음은 0, 조금 많음은 4, 매우 많음은 10`
    - 크기 차이가 머신러닝 결과에 영향을 끼칠 수 있음

#### 명목형 자료 변환하기 - 수치 맵핑
```
import pandas as pd

# 데이터를 읽어 옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Sex'].head())

'''
변환 전: 
 0      male
1    female
2    female
3    female
4      male
Name: Sex, dtype: object
'''

"""
1. replace를 사용하여 male -> 0, female -> 1로 변환합니다.
"""
titanic = titanic.replace({'male':0, 'female':1})

# 변환한 성별 데이터를 출력합니다.
print('\n변환 후: \n',titanic['Sex'].head())

'''
변환 후: 
0    0
1    1
2    1
3    1
4    0
Name: Sex, dtype: int64
'''
```

#### 명목형 자료 변환하기 - 더미 방식
```
import pandas as pd
   
# 데이터를 읽어 옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Embarked'].head())

'''
변환 전: 
0    S
1    C
2    S
3    S
4    S
Name: Embarked, dtype: object
'''

"""
1. get_dummies를 사용하여 변환합니다.
"""
dummies = pd.get_dummies(titanic[['Embarked']])

# 변환한 Embarked 데이터를 출력합니다.
print('\n변환 후: \n',dummies.head())

'''
변환 후: 
    Embarked_C  Embarked_Q  Embarked_S
0           0           0           1
1           1           0           0
2           0           0           1
3           0           0           1
4           0           0           1
'''
```

### 수치형 자료 전 처리
- **수치형 자료** : 크기를 갖는 수치형 값으로 이루어진 데이터
- 머신러닝의 입력으로 바로 사용할 수 있으나, 모델의 성능을 높이기 위해 데이터 변환이 필요

#### 대표적인 수치형 자료 변환 방식
- **스케일링(Scaling)**
  - 변수 값의 범위 및 크기를 변환하는 방식
    - 변수(feature) 간의 범위가 차이가 나면 사용 
  - **정규화(Normalization)**
    - **최소-최대 정규화(Min-Max Normalization)**
      - ![image](https://user-images.githubusercontent.com/61646760/140040796-b7d0b27a-f606-4be3-ba7d-bb80144e4ea3.png)
      - 데이터를 모두 최소 0, 최대 1의 값으로 변환하는 것 (서로 다른 Feature의 scale를 통일하기 위해 변환)
  - **표준화(Standardization)**
    - **Z-Score Scaling**
      - ![image](https://user-images.githubusercontent.com/61646760/140042501-6d556634-e829-4282-83b4-7154fcefc7fb.png)
      - 데이터를 평균이 0, 분산이 1인 가우시안 정규분포로 만들어 주는 것
- 범주화
  - 변수의 값보다 범주가 중요한 경우 사용
  - `예) 시험 점수가 평균 이상이면 1, 미만이면 0으로 범주화 (점수 자체는 중요하지 않음)`

#### 수치형 자료 변환하기 - 정규화

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/140040796-b7d0b27a-f606-4be3-ba7d-bb80144e4ea3.png">
</p>

- 수치형 자료의 경우 다른 수치형 자료와 범위를 맞추기 위해 정규화 또는 표준화를 수행

```
import pandas as pd

"""
1. 정규화를 수행하는 함수를 구현합니다.
"""
def normal(data):
    
    data = (data - data.min()) / (data.max() - data.min())
    
    return data

# 데이터를 읽어 옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Fare'].head())

'''
변환 전: 
0     7.2500
1    71.2833
2     7.9250
3    53.1000
4     8.0500
Name: Fare, dtype: float64
'''

# normal 함수를 사용하여 정규화합니다.
Fare = normal(titanic['Fare'])

# 변환한 Fare 데이터를 출력합니다.
print('\n변환 후: \n',Fare.head())

'''
변환 후: 
0    0.014151
1    0.139136
2    0.015469
3    0.103644
4    0.015713
Name: Fare, dtype: float64
'''
```

#### 수치형 자료 변환하기 - 표준화

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/140042501-6d556634-e829-4282-83b4-7154fcefc7fb.png">
</p>

- 수치형 자료의 경우 다른 수치형 자료와 범위를 맞추기 위해 정규화 또는 표준화를 수행

```
import pandas as pd

"""
1. 표준화를 수행하는 함수를 구현합니다.
"""
def standard(data):
    
    data = (data - data.mean()) / data.std()
    
    return data
    
# 데이터를 읽어 옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Fare'].head())

'''
변환 전: 
0     7.2500
1    71.2833
2     7.9250
3    53.1000
4     8.0500
Name: Fare, dtype: float64
'''

# standard 함수를 사용하여 표준화합니다.
Fare = standard(titanic['Fare'])

# 변환한 Fare 데이터를 출력합니다.
print('\n변환 후: \n',Fare.head())

'''
변환 후: 
0   -0.502163
1    0.786404
2   -0.488580
3    0.420494
4   -0.486064
Name: Fare, dtype: float64
'''
```

### 데이터 정제 및 분리하기
#### 결측값(Missing data) 처리하기
값이 없는 경우
- 일반적인 머신러닝 모델의 입력 값으로 결측값을 사용할 수 없음
- 따라서 **Null, None, NaN** 등의 결측값을 처리해야 함
- 이것이 곧 '데이터 정제'의 과정임
- 대표적인 결측값 처리 방식
  1. 결측값이 존재하는 **샘플 삭제**
  2. 결측값이 많이 존재하는 **변수 삭제**
  3. 결측값을 **다른 값으로 대체**

#### 이상치(Outlier) 처리하기
극단치. 다른 데이터보다 아주 작은 값이나 아주 큰 값
- 이상치가 있으면, 모델의 성능을 저하할 수 있음
- 이상치는 일반적으로 전 처리 과정에서 제거하며, 어떤 값이 이상치인지 판단하는 기준이 중요함
- 이상치 판단 기준 방법
  1. 통계 지표(카이제곱 검정, IQR 지표 등)를 사용하여 판단
  2. 데이터 분포를 보고 직접 판단
  3. 머신러닝 기법을 사용하여 이상치 분류

#### 데이터 분리가 필요한 이유
- 머신러닝 모델을 평가하기 위해서는 학습에 사용하지 않은 평가용 데이터가 필요
- 약 7:3 ~ 8:2 비율로 학습용, 평가용 데이터를 분리함

#### 지도학습 데이터 분리
지도학습의 경우 feature 데이터와 label 데이터를 분리하여 저장
- **Feature 데이터** : label을 예측하기 위한 입력 값
- **Label 데이터** : 예측해야 할 대상이 되는 데이터
  - `예) '공부 시간(feature)'에 따른 '시험 점수(label)'`
  - `예) titanic 데이터에서 feature data는 나이, 가족 정보, 표 가격 등이고 label data는 '생존 여부'`
- scikitlearn의 **`train_test_split()`** 사용

#### 결측값 처리하기
결측값이 있는 데이터는 일반적으로 머신러닝의 입력으로 사용할 수 없습니다. 그렇기에 데이터 전 처리 과정에서는 삭제 또는 대체 방식으로 결측값을 처리합니다.
- **`drop(columns = ['feature'])`** : pandas의 DataFrame에서 특정 변수(columns)를 삭제하기 위해 사용
- **`dropna()`** : DataFrame에서 결측값이 있는 샘플을 제거하기 위해서 사용 (NaN 제거)

```
import pandas as pd
    
# 데이터를 읽어 옵니다.
titanic = pd.read_csv('./data/titanic.csv')
# 변수별 데이터 수를 확인하여 결측값이 어디에 많은지 확인합니다.
print(titanic.info(),'\n')  # cabin feature가 204/891로 결측치가 제일 많음

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None 
'''

"""
1. Cabin 변수를 제거합니다.
"""
titanic_1 = titanic.drop(columns = ['Cabin'])
# Cabin 변수를 제거 후 결측값이 어디에 남아 있는지 확인합니다.
print('Cabin 변수 제거')
print(titanic_1.info(),'\n')

'''
Cabin 변수 제거
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 11 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(4)
memory usage: 76.6+ KB
None 
'''

"""
2. 결측값이 존재하는 샘플 제거합니다.
"""
titanic_2 = titanic_1.dropna()
# 결측값이 존재하는지 확인합니다.
print('결측값이 존재하는 샘플 제거')
print(titanic_2.info())

'''
결측값이 존재하는 샘플 제거
<class 'pandas.core.frame.DataFrame'>
Int64Index: 712 entries, 0 to 890
Data columns (total 11 columns):
PassengerId    712 non-null int64
Survived       712 non-null int64
Pclass         712 non-null int64
Name           712 non-null object
Sex            712 non-null object
Age            712 non-null float64
SibSp          712 non-null int64
Parch          712 non-null int64
Ticket         712 non-null object
Fare           712 non-null float64
Embarked       712 non-null object
dtypes: float64(2), int64(5), object(4)
memory usage: 66.8+ KB
None
'''
```

#### 이상치 처리하기
이상치가 존재하는 데이터를 머신러닝에 사용하게 된다면 성능 저하를 야기할 수 있습니다. 그렇기에 데이터 전 처리 과정에서는 이상치를 판별하고 처리합니다.

```
import pandas as pd
import numpy as np

# 데이터를 읽어 옵니다.
titanic = pd.read_csv('./data/titanic.csv')

# Cabin 변수를 제거합니다.
titanic_1 = titanic.drop(columns=['Cabin'])

# 결측값이 존재하는 샘플을 제거합니다.
titanic_2 = titanic_1.dropna()

# (Age 값 - 내림 Age 값)이 0보다 크다면 소수점을 갖는 데이터로 분류합니다.
outlier = titanic_2[titanic_2['Age']-np.floor(titanic_2['Age']) > 0 ]['Age']

print('소수점을 갖는 Age 변수 이상치')
print(outlier)
print('이상치 처리 전 샘플 개수: %d' %(len(titanic_2)))
print('이상치 개수: %d' %(len(outlier)))

'''
소수점을 갖는 Age 변수 이상치
57     28.50
78      0.83
111    14.50
116    70.50
122    32.50
123    32.50
148    36.50
152    55.50
153    40.50
203    45.50
227    20.50
296    23.50
305     0.92
331    45.50
469     0.75
525    40.50
644     0.75
676    24.50
735    28.50
755     0.67
767    30.50
803     0.42
814    30.50
831     0.83
843    34.50
Name: Age, dtype: float64
이상치 처리 전 샘플 개수: 712
이상치 개수: 25
'''

"""
1. 이상치를 처리합니다.
"""
titanic_3 = titanic_2[titanic_2['Age'] - np.floor(titanic_2['Age']) == 0]  # Age 값 - Age 내림 값이 0이면 이상치 아님
print('이상치 처리 후 샘플 개수: %d' %(len(titanic_3)))                      # 이상치 처리 후 샘플 개수: 687
```

#### 데이터 분리하기
- `train_test_split()` : sklearn 라이브러리의 학습용, 평가용 데이터 분리 메소드
  - `X_train, X_test, y_train, y_test = train_test_split(feature 데이터, label 데이터, test_size = 0~1 값, random_state = 랜덤 시드 값)`

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')

# Cabin 변수를 제거합니다.
titanic_1 = titanic.drop(columns=['Cabin'])

# 결측값이 존재하는 샘플을 제거합니다.
titanic_2 = titanic_1.dropna()

# 이상치를 처리합니다.
titanic_3 = titanic_2[titanic_2['Age']-np.floor(titanic_2['Age']) == 0 ]
print('전체 샘플 데이터 개수: %d' %(len(titanic_3)))  # 전체 샘플 데이터 개수: 687

"""
1. feature 데이터와 label 데이터를 분리합니다.
"""
X = titanic_3.drop(columns = ['Survived'])
y = titanic_3['Survived']
print('X 데이터 개수: %d' %(len(X)))  # X 데이터 개수: 687
print('y 데이터 개수: %d' %(len(y)))  # y 데이터 개수: 687

"""
2. 학습용, 평가용 데이터로 분리합니다.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 분리한 데이터의 개수를 출력합니다.
print('학습용 데이터 개수: %d' %(len(X_train)))  # 학습용 데이터 개수: 480
print('평가용 데이터 개수: %d' %(len(X_test)))   # 평가용 데이터 개수: 207
```


## 03. 지도학습 - 회귀
### 회귀 개념 알아보기
- **회귀 분석(Regression analysis)**
  - 데이터를 **가장 잘 설명하는 모델**을 찾아 입력값에 따른 미래 결과값을 예측하는 알고리즘
    - 관찰된 연속형 변수들에 대해 두 변수 사이의 모형을 구한뒤 적합도를 측정해 내는 분석 방법 (위키백과)
  - `예) 평균 기온에 따른 아이스크림 판매량`
    - ![image](https://user-images.githubusercontent.com/61646760/140511678-8a27bee9-ee28-4d43-ae15-39f67aa579eb.png) : 평균 기온, ![image](https://user-images.githubusercontent.com/61646760/140511705-b96723ef-840e-47d7-951c-2d49b268580e.png) : 아이스크림 판매량 (데이터)
    - ![image](https://user-images.githubusercontent.com/61646760/140510646-e44461c8-c449-43c3-ad6d-2f0bb3cd56c7.png) (가정)
      - 이 직선이 곧 데이터를 가장 잘 설명(= 실제에 근사)하는 모델 
      - 각 데이터의 실제 값과 모델이 예측하는 값의 차이를 최소한으로 하는 선을 찾자.
        - 즉, 적절한 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png)와 ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png)을 찾으면 된다.

### 단순 선형 회귀
- **단순 선형 회귀(Simple Linear Regression)**
  - 데이터를 설명하는 모델을 직선 형태로 가정
    - ![image](https://user-images.githubusercontent.com/61646760/140510646-e44461c8-c449-43c3-ad6d-2f0bb3cd56c7.png)
    - 직선을 구성하는 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png) (y 절편)와 ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) (기울기)를 구해야 함
  - 특징
    - 가장 기초적이나 여전히 많이 사용되는 알고리즘
    - 입력값이 1개인 경우에만 적용이 가능함
    - 입력값과 결과값의 관계를 알아보는 데 용이함
    - 입력값이 결과값에 얼마나 영향을 미치는지 알 수 있음
    - 두 변수 간의 관계를 직관적으로 해석하고자 하는 경우 활용

#### 단순 선형 회귀 모델의 손실 함수
- **손실 함수(Loss function)**
  - 실제 값과 예측 값 차이의 제곱의 합
    - ![image](https://user-images.githubusercontent.com/61646760/140515451-9b2c9f1c-def4-4090-93fd-d52a7066ee1d.png)
    - 손실 함수가 작을수록 좋은 모델이다.
  - 손실 함수 줄이기
    - ![image](https://user-images.githubusercontent.com/61646760/140517279-bd8734c9-c3e0-46d7-9a62-c175adc4778f.png)
      - ![image](https://user-images.githubusercontent.com/61646760/140517368-aad92e73-10c4-41bf-9571-7f88b6b77afd.png) (arguments of minimum) : 함수 ![image](https://user-images.githubusercontent.com/61646760/140516833-752063c9-6072-4a16-81d8-9d0c50d3d555.png)를 최솟값으로 만드는 정의역(![image](https://user-images.githubusercontent.com/61646760/140516872-37a9d24e-4c30-4fb0-ad9e-182726014faa.png))의 값
    - ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png) (y 절편), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) (기울기) 값을 조절하여 Loss 함수의 크기를 작게 할 수 있다.
  - Loss 함수의 크기를 작게 하는 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png) (y 절편), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) (기울기)를 찾는 방법
      1. **Gradient descent (경사 하강법)**
      2. **Normal equation (least squres)**

#### 단순 선형 회귀 모델의 경사 하강법
- **경사 하강법(Gradient descent)**
  - 경사 하강법은 계산 한번으로 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png)을 구하는 것이 아니라 초기값에서 점진적으로 구하는 방식
  - ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) 값을 Loss 함수 값이 작아지게 계속 업데이트하는 방법
    1. ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) 값을 랜덤하게 초기화
    2. 현재 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) 값으로 Loss 값 계산
    3. 현재 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) 값을 어떻게 변화해야 Loss 값을 줄일 수 있는지 알 수 있는 **Gradient 값** 계산
    4. Gradient 값을 활용하여 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) 값 업데이트
    5. Loss 값의 차이가 거의 없어질 때까지 2~4번 과정을 반복 (Loss 값과 차이가 줄어들면 Gradient 값도 작아짐) 

![image](https://user-images.githubusercontent.com/61646760/140540230-183b469c-3d54-4fdb-9ff0-5021a7cf0fc0.png)

#### 단순 선형 회귀 분석하기 - 데이터 전 처리
- `LinearRegression` : 기계학습 라이브러리 scikit-learn의 선형 회귀를 위한 클래스
  - `from sklearn.linear_model import LinearRegression`

```
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

"""
1. X의 형태를 변환하여 train_X에 저장합니다.
"""
train_X = pd.DataFrame(X, columns = ['X'])  # X 데이터를 column 명이 X인 DataFrame으로 변환

"""
2. Y의 형태를 변환하여 train_Y에 저장합니다.
"""
train_Y = pd.Series(Y)  # 리스트 Y를 Series 형식으로 변환

# 변환된 데이터를 출력합니다.
print('전 처리한 X 데이터: \n {}'.format(train_X))
print('전 처리한 X 데이터 shape: {}\n'.format(train_X.shape))

'''
전 처리한 X 데이터: 
           X
0  8.701538
1  3.908258
2  1.893624
3  3.287300
4  7.393330
5  2.989846
6  2.257572
7  9.844507
8  9.945895
9  5.483216
전 처리한 X 데이터 shape: (10, 1)
'''

print('전 처리한 Y 데이터: \n {}'.format(train_Y))
print('전 처리한 Y 데이터 shape: {}'.format(train_Y.shape))

'''
전 처리한 Y 데이터: 
0    5.644131
1    3.758766
2    3.872333
3    4.409904
4    6.438450
5    4.028278
6    2.261060
7    7.157690
8    6.290974
9    5.196929
dtype: float64
전 처리한 Y 데이터 shape: (10,)
'''
```

#### 단순 선형 회귀 분석하기 - 학습하기
바로 위에서 전 처리한 데이터를 `LinearRegression` 모델에 입력하여 학습을 수행해 보자.

- LinearRegression (sklearn) 사용법
  1. `LinearRegression`을 사용하기 위해서는 우선 해당 모델 객체를 불러와 초기화해야 한다.
      - `lrmodel = LinearRegression()`
  2. 모델 초기화를 수행했다면 전 처리된 데이터를 사용하여 학습을 수행할 수 있다. 아래 코드와 같이 `fit` 함수에 학습에 필요한 데이터를 입력하여 학습을 수행한다.
      - `lrmodel.fit(train_X, train_Y)`
  3. LinearRegression의 ![image](https://user-images.githubusercontent.com/61646760/140510991-70ef4523-52e2-4360-87eb-093d55107db8.png), ![image](https://user-images.githubusercontent.com/61646760/140511012-753ddb9d-d758-4522-9853-280250a50d2a.png) 값을 구하기 위해서는 아래 코드를 사용하면 된다.
      ```
      beta_0 = lrmodel.intercept_
      beta_1 = lrmodel.coef_[0]
      ```
```
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# import elice_utils
# eu = elice_utils.EliceUtils()


X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

train_X = pd.DataFrame(X, columns=['X'])
train_Y = pd.Series(Y)

"""
1. 모델을 초기화합니다.
"""
lrmodel = LinearRegression()

"""
2. train_X, train_Y 데이터를 학습합니다.
"""
lrmodel.fit(train_X, train_Y)


# 학습한 결과를 시각화하는 코드입니다.
plt.scatter(X, Y) 
plt.plot([0, 10], [lrmodel.intercept_, 10 * lrmodel.coef_[0] + lrmodel.intercept_], c='r') 
plt.xlim(0, 10) 
plt.ylim(0, 10) 
plt.title('Training Result')
plt.savefig("test.png") 
# eu.send_image("test.png")
```
![image](https://user-images.githubusercontent.com/61646760/140742957-59635673-f019-4de9-adc0-5e56a912aa9d.png)

#### 단순 선형 회귀 분석하기 - 예측하기
바로 위에서 학습한 모델을 바탕으로 예측 값을 구해 봅시다.
- `predict()` : `LinearRegression`으로 예측을 할 때 사용하는 메서드. `predict` 함수는 DataFrame 또는 numpy array인 X 데이터에 대한 예측값을 리스트로 출력한다.
  - `pred_X = lrmodel.predict(X)`

```
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

train_X = pd.DataFrame(X, columns=['X'])
train_Y = pd.Series(Y)

# 모델을 트레이닝합니다.
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

"""
1. train_X에 대해서 예측합니다.
"""
pred_X = lrmodel.predict(train_X)
print('train_X에 대한 예측값 : \n{}\n'.format(pred_X))

'''
train_X에 대한 예측값 : 
[6.2546398  4.18978504 3.32191889 3.92228833 5.6910886  3.79415077
 3.47870087 6.74700964 6.7906856  4.86824749]
'''

print('실제값 : \n{}'.format(train_Y))

'''
실제값 : 
0    5.644131
1    3.758766
2    3.872333
3    4.409904
4    6.438450
5    4.028278
6    2.261060
7    7.157690
8    6.290974
9    5.196929
dtype: float64
'''
```

### 다중 선형 회귀
- **다중 선형 회귀((Multiple Linear Regression)**
  - 입력값 ![image](https://user-images.githubusercontent.com/61646760/140957356-e421f52e-0067-4cd6-921d-3f68e1fd5bdb.png)가 여러 개(2개 이상)인 경우 활용할 수 있는 회귀 알고리즘
    - 여러 개의 입력값(![image](https://user-images.githubusercontent.com/61646760/140957356-e421f52e-0067-4cd6-921d-3f68e1fd5bdb.png))으로 결괏값(![image](https://user-images.githubusercontent.com/61646760/140957508-73a8350e-1b1f-4d46-9b5d-29ea92c24d54.png))을 예측하고자 하는 경우
    - 각 개별 ![image](https://user-images.githubusercontent.com/61646760/140958263-a2a0679a-c66e-43a0-8173-cc264dbe7d98.png)에 해당하는 최적의 ![image](https://user-images.githubusercontent.com/61646760/140958332-84f665c2-7bbb-4071-ba41-1d8bc8f178f2.png)를 찾아야 함
  - 다중 선형 회귀 모델
    - ![image](https://user-images.githubusercontent.com/61646760/140958761-d525039a-3bbc-4e0a-882a-8d0838e6cc53.png)
  - 특징
    - 여러 개의 입력값과 결괏값 간의 관계 확인 가능
    - 어떤 입력값이 결괏값에 어떠한 영향을 미치는지 알 수 있음
    - 여러 개의 입력값 간의 상관관계가 높을 경우 결과에 대한 신뢰성을 잃을 가능성이 있음

#### 다중 선형 회귀 모델의 손실 함수
  - **손실 함수(Loss function)**
    - 단순 선형 회귀(Simple linear regression)와 마찬가지로 Loss 함수는 **입력값과 실제값 차이의 제곱의 합**으로 정의함
      - 마찬가지로 ![image](https://user-images.githubusercontent.com/61646760/140959866-3d8c3475-a0c5-43dc-ac6c-811b14ab3a43.png) 값을 조절하여 Loss 함수의 크기를 작게 함
    - ![image](https://user-images.githubusercontent.com/61646760/140960865-543ec942-202f-420d-ba87-2a816798ed69.png)

#### 다중 선형 회귀 모델의 경사 하강법
- **경사 하강법(Gradient descent)**
  - ![image](https://user-images.githubusercontent.com/61646760/140962141-3a565d9e-b5de-4136-9b4e-0b82a6f8e968.png)값을 Loss 함수 값이 작아지게 계속 업데이트하는 방법
    1. ![image](https://user-images.githubusercontent.com/61646760/140962218-22234c8c-5147-4601-b0b3-5af6ffaa1fc6.png) 값을 랜덤하게 초기화
    2. 현재 ![image](https://user-images.githubusercontent.com/61646760/140962218-22234c8c-5147-4601-b0b3-5af6ffaa1fc6.png) 값으로 Loss 값 계산
    3. 현재 ![image](https://user-images.githubusercontent.com/61646760/140962218-22234c8c-5147-4601-b0b3-5af6ffaa1fc6.png) 값을 어떻게 변화해야 Loss 값을 줄일 수 있는지 알 수 있는 Gradient 값 계산
    4. Gradient 값을 활용하여 ![image](https://user-images.githubusercontent.com/61646760/140962218-22234c8c-5147-4601-b0b3-5af6ffaa1fc6.png) 값 업데이트
    5. Loss 값의 차이가 거의 없어질 때까지 2~4번 과정을 반복(Loss 값과 차이가 줄어들면 Gradient 값도 작아짐)

#### 다중 회귀 분석하기 - 데이터 전 처리
- <strong>다중 회귀 분석(Multiple Linear Regression)</strong>은 데이터의 여러 변수(features) ![image](https://user-images.githubusercontent.com/61646760/140957356-e421f52e-0067-4cd6-921d-3f68e1fd5bdb.png)를 이용해 결과 ![image](https://user-images.githubusercontent.com/61646760/140957508-73a8350e-1b1f-4d46-9b5d-29ea92c24d54.png)를 예측하는 모델
- 아래와 같이 `FB`, `TV`, `Newspaper` 광고에 대한 비용 대비 `Sales` 데이터가 주어졌을 때, 이를 다중 회귀 분석으로 분석해 보자.
  ![image](https://user-images.githubusercontent.com/61646760/142372861-1edc2251-0f05-429a-9c89-4123e88e3d9c.png)
- 우선 데이터를 전 처리 하기 위해서 3개의 변수를 갖는 feature 데이터와 Sales 변수를 label 데이터로 분리하고 학습용, 평가용 데이터로 나눠 보자.

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/Advertising.csv")

print('원본 데이터 샘플 :')
print(df.head(),'\n')

'''
원본 데이터 샘플 :
   Unnamed: 0     FB    TV  Newspaper  Sales
0           1  230.1  37.8       69.2   22.1
1           2   44.5  39.3       45.1   10.4
2           3   17.2  45.9       69.3    9.3
3           4  151.5  41.3       58.5   18.5
4           5  180.8  10.8       58.4   12.9 
'''

# 입력 변수로 사용하지 않는 Unnamed: 0 변수 데이터를 삭제합니다.
df = df.drop(columns=['Unnamed: 0'])

"""
1. Sales 변수는 label 데이터로 Y에 저장하고 나머진 X에 저장합니다.
"""
X = df.drop(columns=['Sales'])
Y = df['Sales']

"""
2. 학습용 평가용 데이터로 분리합니다.
"""
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 전 처리한 데이터를 출력합니다.
print('train_X : ')
print(train_X.head(),'\n')

'''
train_X : 
        FB    TV  Newspaper
79   116.0   7.7       23.1
197  177.0   9.3        6.4
38    43.1  26.7       35.1
24    62.3  12.6       18.3
122  224.0   2.4       15.6 
'''

print('train_Y : ')
print(train_Y.head(),'\n')

'''
train_Y : 
79     11.0
197    12.8
38     10.1
24      9.7
122    11.6
Name: Sales, dtype: float64 
'''

print('test_X : ')
print(test_X.head(),'\n')

'''
test_X : 
        FB    TV  Newspaper
95   163.3  31.6       52.9
15   195.4  47.7       52.9
30   292.9  28.3       43.2
158   11.7  36.9       45.2
128  220.3  49.0        3.2 
'''

print('test_Y : ')
print(test_Y.head())

'''
test_Y : 
95     16.9
15     22.4
30     21.4
158     7.3
128    24.7
Name: Sales, dtype: float64
'''
```

#### 다중 회귀 분석하기 - 학습하기
앞에서 전 처리한 데이터를 바탕으로 다중 선형 회귀 모델을 적용해 보자.

- 다중 선형 회귀 모델의 형태는 아래 수식과 같다.
  - ![image](https://user-images.githubusercontent.com/61646760/142715604-d47855f9-68cd-4238-9d88-9b67251e77f2.png)
  - 여기서 ![image](https://user-images.githubusercontent.com/61646760/142715624-5edc38d5-5f9b-49e7-b35a-f9b62fac9d13.png)은 페이스북, ![image](https://user-images.githubusercontent.com/61646760/142715629-60252968-7266-4946-8899-35791add9e39.png)는 TV, ![image](https://user-images.githubusercontent.com/61646760/142715639-735f4388-a739-4fe6-ab8a-45948099950e.png)은 신문 광고를 의미한다.

- 다중 선형 회귀 또한 선형 회귀 모델과 마찬가지로 `LinearRegression`을 사용할 수 있다.
  - `LinearRegression`의 ![image](https://user-images.githubusercontent.com/61646760/142715716-ae01e5da-546a-4466-bfec-1b020a4e7d4f.png) 등 패러미터 구현
    ```
    lrmodel = LinearRegression()
    lrmodel.intercept_
    lrmodel.coef_[i]
    ```
    - `intercept_` : ![image](https://user-images.githubusercontent.com/61646760/142715774-e80e590e-0d2c-4bcb-986e-ec9419da7d32.png)에 해당하는 값
    - `coef_[i]` : i+1번째 변수에 곱해지는 패러미터 값
- 그럼 이번에는 학습용 데이터를 다중 선형 회귀 모델을 사용하여 학습하고, 학습된 패러미터를 출력해 보자.

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

# print(df.head())
'''
      FB    TV  Newspaper  Sales
0  230.1  37.8       69.2   22.1
1   44.5  39.3       45.1   10.4
2   17.2  45.9       69.3    9.3
3  151.5  41.3       58.5   18.5
4  180.8  10.8       58.4   12.9
'''

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

"""
1. 다중 선형 회귀 모델을 초기화하고 학습합니다
"""
lrmodel = LinearRegression()   # 모델 초기화
lrmodel.fit(train_X, train_Y)  # train data로 학습

"""
2. 학습된 패러미터 값을 불러옵니다
"""
beta_0 = lrmodel.intercept_   # y절편 (기본 판매량)
beta_1 = lrmodel.coef_[0]     # 1번째 변수에 대한 계수 (페이스북)
beta_2 = lrmodel.coef_[1]     # 2번째 변수에 대한 계수 (TV)
beta_3 = lrmodel.coef_[2]     # 3번째 변수에 대한 계수 (신문)

print("beta_0: %f" % beta_0)  # beta_0: 2.979067
print("beta_1: %f" % beta_1)  # beta_1: 0.044730
print("beta_2: %f" % beta_2)  # beta_2: 0.189195
print("beta_3: %f" % beta_3)  # beta_3: 0.002761
```

#### 다중 회귀 분석하기 - 예측하기
위에서 학습한 다중 선형 회귀 모델을 바탕으로 이번엔 새로운 광고 비용에 따른 `Sales` 값을 예측해 보자.

- `predict()` : `LinearRegression`의 예측을 위한 함수
  - `pred_X = lrmodel.predict(X)`

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


# 다중 선형 회귀 모델을 초기화하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)


print('test_X : ')
print(test_X)

'''
test_X : 
        FB    TV  Newspaper
95   163.3  31.6       52.9
15   195.4  47.7       52.9
30   292.9  28.3       43.2
158   11.7  36.9       45.2
128  220.3  49.0        3.2
115   75.1  35.0       52.7
69   216.8  43.9       27.2
170   50.0  11.6       18.4
174  222.4   3.4       13.1
45   175.1  22.5       31.5
66    31.5  24.6        2.2
182   56.2   5.7       29.7
165  234.5   3.4       84.8
78     5.4  29.9        9.4
186  139.5   2.1       26.6
177  170.2   7.8       35.2
56     7.3  28.1       41.4
152  197.6  23.3       14.2
82    75.3  20.3       32.5
68   237.4  27.5       11.0
124  229.5  32.3       74.2
16    67.8  36.6      114.0
148   38.0  40.3       11.9
93   250.9  36.5       72.3
65    69.0   9.3        0.9
60    53.5   2.0       21.4
84   213.5  43.0       33.8
67   139.3  14.5       10.2
125   87.2  11.8       25.9
132    8.4  27.2        2.1
9    199.8   2.6       21.2
18    69.2  20.5       18.3
55   198.9  49.4       60.0
75    16.9  43.7       89.4
150  280.7  13.9       37.0
104  238.2  34.3        5.3
135   48.3  47.0        8.5
137  273.7  28.9       59.7
164  117.2  14.7        5.4
76    27.5   1.6       20.7
'''

"""
1. test_X에 대해서 예측합니다.
"""
pred_X = lrmodel.predict(test_X)
print('test_X에 대한 예측값 : \n{}\n'.format(pred_X))

'''
test_X에 대한 예측값 : 
[16.4080242  20.88988209 21.55384318 10.60850256 22.11237326 13.10559172
 21.05719192  7.46101034 13.60634581 15.15506967  9.04831992  6.65328312
 14.34554487  8.90349333  9.68959028 12.16494386  8.73628397 16.26507258
 10.27759582 18.83109103 19.56036653 13.25103464 12.33620695 21.30695132
  7.82740305  5.80957448 20.75753231 11.98138077  9.18349576  8.5066991
 12.46646769 10.00337695 21.3876709  12.24966368 18.26661538 20.13766267
 14.05514005 20.85411186 11.0174441   4.56899622]
'''

# 새로운 데이터 df1을 정의합니다
df1 = pd.DataFrame(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]), columns=['FB', 'TV', 'Newspaper'])
print('df1 : ')
print(df1)

'''
df1 : 
   FB  TV  Newspaper
0   0   0          0
1   1   0          0
2   0   1          0
3   0   0          1
4   1   1          1
'''

"""
2. df1에 대해서 예측합니다.
"""
pred_df1 = lrmodel.predict(df1)
print('df1에 대한 예측값 : \n{}'.format(pred_df1))

'''
df1에 대한 예측값 : 
[2.97906734 3.02379686 3.16826239 2.98182845 3.21575302]
'''
```

### 회귀 평가 지표
- 모델링 이후에는 해당 모델이 얼마나 목표를 잘 달성했는지 그 정도를 평가해야 한다.
  - 실제 값과 모델이 예측하는 값의 차이에 기반한 평가 방법 사용
    - 예) ![image](https://user-images.githubusercontent.com/61646760/142759412-84aacf7b-bd95-4071-a264-5f3e8e46d014.png), ![image](https://user-images.githubusercontent.com/61646760/142759437-1d45dec5-2f97-497d-b0d3-5648a1cd73c3.png), ![image](https://user-images.githubusercontent.com/61646760/142759469-0800fe04-7336-4023-a420-705fa33d6723.png), ![image](https://user-images.githubusercontent.com/61646760/142759471-dc4bdb78-9b84-4a4b-8f00-8d4ec18794bc.png), ![image](https://user-images.githubusercontent.com/61646760/142759478-c9256bbc-7d68-4353-8486-07eb139bc986.png)

#### RSS - 단순 오차

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/142760068-08dc44aa-2393-40fc-8206-469b206a11ae.png" />
</p>

- **잔차 제곱 합 (RSS: residual sum of squares)**
  - 실제 값과 예측 값의 단순 오차 제곱 합 
  - 값이 작을수록 모델의 성능이 높음 
  - 전체 데이터에 대한 실제 값과 예측하는 값의 오차 제곱의 총합
- RSS 특징
  - 가장 간단한 평가 방법으로 직관적인 해석이 가능함
  - 그러나 오차를 그대로 이용하기 때문에 입력값의 **크기에 의존적**임
  - 절대적인 값과 비교가 불가능함

#### MSE, MAE - 절대적인 크기에 의존한 지표

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/142760333-4ecafd8d-d7d0-499a-882d-54a8a119ab0a.png" /><br>
  <img src="https://user-images.githubusercontent.com/61646760/142760435-ef233285-b219-4b67-96a7-490740a7003c.png" />
</p>

- **평균 제곱 오차 (MSE: Mean Squared Error)**
  - RSS에서 데이터 수만큼 나눈 값
  - 작을수록 모델의 성능이 높다고 평가할 수 있음
- **평균 절대 오차 (MAE: Mean Absolute Error)**
  - 평균 절댓값 오차
  - 실제 값과 예측값의 오차의 절댓값의 평균
  - 작을수록 모델의 성능이 높다고 평가할 수 있음
- MSE, MAE 특징
  - **MSE** : 이상치(Outlier) 즉, 데이터들 중 크게 떨어진 값에 민감함 
  - **MAE** : 변동성이 큰 지표와 낮은 지표를 같이 예측할 시 유용 
  - 가장 간단한 평가 방법들로 직관적인 해석이 가능함 
  - 그러나 평균을 그대로 이용하기 때문에 입력 값의 크기에 의존적임 
  - 절대적인 값과 비교가 불가능함

#### R-squared - 결정 계수

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/142835658-c586442c-0709-4651-be33-2f40603126d4.png" />
</p>

- **결정 계수(![image](https://user-images.githubusercontent.com/61646760/142835228-38f1151c-22ba-4ad5-b8d7-15b8b0664685.png): coefficient of determination)**
  - 회귀 모델의 설명력을 표현하는 지표
  - 1에 가까울수록 높은 성능의 모델이라고 해석할 수 있음
    - ![image](https://user-images.githubusercontent.com/61646760/142835778-6a263452-12e5-416c-b2a8-835f481a3946.png) (total sum of squares)는 데이터 평균 값(![image](https://user-images.githubusercontent.com/61646760/142835971-0fb7ebf0-85cd-4a52-abe4-ee67a805f2ea.png))과 실제 값(![image](https://user-images.githubusercontent.com/61646760/142836027-edc66423-27c4-4874-b951-44b9b3d929b4.png)) 차이의 제곱  
      ![image](https://user-images.githubusercontent.com/61646760/142836736-379942fe-236a-44e2-bb41-8c8a3086e935.png)  
      ![image](https://user-images.githubusercontent.com/61646760/142836946-8afdbc19-385e-4455-b340-d0b2b0280d4b.png)  
        ![image](https://user-images.githubusercontent.com/61646760/142837658-7271caa9-919a-4e9b-ad57-f27753031457.png)
    - ![image](https://user-images.githubusercontent.com/61646760/142835228-38f1151c-22ba-4ad5-b8d7-15b8b0664685.png) 특징
      - 오차가 없을수록 1에 가까운 값을 가짐
      - 값이 0인 경우, 데이터의 평균 값을 출력하는 직선 모델을 의미함
      - 음수 값이 나온 경우, 평균값 예측보다 성능이 좋지 않음
    - ![image](https://user-images.githubusercontent.com/61646760/142838432-328771fa-e7d5-42fa-8a34-5764a585133f.png)
      - **total sum of squares (SST)**
        - 개별 y의 편차 제곱의 합  
          ![image](https://user-images.githubusercontent.com/61646760/142839698-5823f425-7ec8-484d-8e1e-f1f921914289.png)
      - **explained sum of squares (SSE)**
        - 회귀식 추정 y의 편차 제곱의 합  
          ![image](https://user-images.githubusercontent.com/61646760/142840396-5b02b90c-018a-4d34-9bba-a1ff0f00f09a.png)
        - 경우에 따라 SSR(Regression Sum of Squares)로 표현하기도 함
      - **residual sum of squares (SSR)**
        - 잔차(residual)의 제곱의 합  
          ![image](https://user-images.githubusercontent.com/61646760/142840862-49781907-db39-45c4-aedc-55bc0f53829a.png) 
        - 경우에 따라 SSE(Error Sum of Squares)로 표현하기도 함
      - **squared-R (![image](https://user-images.githubusercontent.com/61646760/142839092-d7a1a1af-69a9-4463-93e1-bab7b62c9484.png))**
        - 총 변동 중에 설명된 변동의 비율  
          ![image](https://user-images.githubusercontent.com/61646760/142841179-f33fb439-1149-4906-9418-4b0d11dd4361.png)

#### 회귀 알고리즘 평가 지표 - MSE, MAE
앞에 이어 `Sales` 예측 모델의 성능을 평가하기 위해서 다양한 회귀 알고리즘 평가 지표를 사용하여 비교해 보자.

이번 실습에서는 학습용 및 평가용 데이터에 대해서 MSE와 MAE을 계산해 보겠다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/142976680-752e8f73-2be7-4808-b5c6-22f261b9e0af.png" /><br>
  <img src="https://user-images.githubusercontent.com/61646760/142976743-a2f65b8c-5d13-49e0-85c0-d9666134ba6e.png" />
</p>

MSE와 MAE는 위와 같이 정의할 수 있고 sklearn 라이브러리 함수를 통하여 쉽게 구할 수 있다. (![image](https://user-images.githubusercontent.com/61646760/142907687-86aa68bc-6ba8-41e2-a4c7-f6df94183369.png)은 전체 샘플의 개수를 의미함)
- MSE, MAE 평가 지표를 계산하기 위한 sklearn 함수
  - `mean_squared_error(y_true, y_pred)`
    - MSE 값 계산하기
    - `from sklearn.metrics import mean_squared_error`
  - `mean_absolute_error(y_true, y_pred)`
    - MAE 값 계산하기
    - `from sklearn.metrics import mean_absolute_error` 

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


# 다중 선형 회귀 모델을 초기화하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)


# train_X 의 예측값을 계산합니다
pred_train = lrmodel.predict(train_X)

"""
1. train_X 의 MSE, MAE 값을 계산합니다
"""
MSE_train = mean_squared_error(train_Y, pred_train)
MAE_train = mean_absolute_error(train_Y, pred_train)
print('MSE_train : %f' % MSE_train)  # MSE_train : 2.705129
print('MAE_train : %f' % MAE_train)  # MAE_train : 1.198468

# test_X 의 예측값을 계산합니다
pred_test = lrmodel.predict(test_X)

"""
2. test_X 의 MSE, MAE 값을 계산합니다
"""
MSE_test = mean_squared_error(test_Y, pred_test)
MAE_test = mean_absolute_error(test_Y, pred_test)
print('MSE_test : %f' % MSE_test)  # MSE_test : 3.174097
print('MAE_test : %f' % MAE_test)  # MAE_test : 1.460757
```

#### 회귀 알고리즘 평가 지표 - R2
앞에 이어 `Sales` 예측 모델의 성능을 평가하기 위해서 다양한 회귀 알고리즘 평가 지표를 사용하여 비교해 보자.

이번 실습에서는 학습용 및 평가용 데이터에 대해 R2 score를 계산해 보겠다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/142977821-c348dc4d-6f6a-43d9-a73b-4303d2f99b0a.png" /><br>
  <img src="https://user-images.githubusercontent.com/61646760/142977844-a60c50f7-eb40-437a-a029-172bd6a51df0.png" />
</p>

R2 score는 위와 같이 정의할 수 있고 sklearn 라이브러리 함수를 통하여 쉽게 구할 수 있다. (![image](https://user-images.githubusercontent.com/61646760/142907687-86aa68bc-6ba8-41e2-a4c7-f6df94183369.png)은 전체 샘플의 개수를 의미함)

- R2 평가 지표를 계산하기 위한 sklearn 함수
  - `r2_score(y_true, y_pred)`
    - R2 score 값 계산하기
    - `from sklearn.metrics import r2_score`

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


# 다중 선형 회귀 모델을 초기화 하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)


# train_X 의 예측값을 계산합니다
pred_train = lrmodel.predict(train_X)

"""
1. train_X 의 R2 값을 계산합니다
"""
R2_train = r2_score(train_Y, pred_train)
print('R2_train : %f' % R2_train)  # R2_train : 0.895701

# test_X 의 예측값을 계산합니다
pred_test = lrmodel.predict(test_X)

"""
2. test_X 의 R2 값을 계산합니다
"""
R2_test = r2_score(test_Y, pred_test)
print('R2_test : %f' % R2_test)  # R2_test : 0.899438
```

## 04. 지도학습 - 분류
### 분류 개념 알아보기
- **분류(Classification)**
  - 주어진 입력 값이 어떤 클래스에 속할지에 대한 결괏값을 도출하는 알고리즘
    - `예) 풍속 4m/s를 기준으로 항공 지연/지연 없음 나누기`  
      ![image](https://user-images.githubusercontent.com/61646760/143068295-b0c06cea-ee9f-4568-aa0a-550573eab5f2.png)
  - 다양한 분류 알고리즘이 존재하며, 예측 목표와 데이터 유형에 따라 다양한 머신러닝 모델 적용
    - **트리 구조 기반** : 의사 결정 트리, 랜덤 포레스트
    - **확률 모델 기반** : 나이브 베이즈 분류기
    - **결정 경계 기반** : 선형 분류기, 로지스틱 회귀 분류기, SVM
    - **신경망** : 퍼셉트론, 딥러닝 모델

### 의사 결정 나무
#### 의사 결정 나무 - 모델 구조
- **의사 결정 나무(Decision Tree)**
  - 스무 고개와 같이 특정 질문들을 통해 정답을 찾아가는 모델
  - 최상단의 뿌리 마디에서 마지막 끝 마디까지 아래 방향으로 진행  
    ![image](https://user-images.githubusercontent.com/61646760/143176725-2ed7a357-13fd-4442-8dcc-5af36eec1f36.png)
  - <strong>중간 마디(Internal Node)</strong>를 통한 분리 기준 추가 가능 [`예)`](https://user-images.githubusercontent.com/61646760/143177517-511b6253-6c97-4c64-8202-74112c1c6475.png) [`예)`](https://user-images.githubusercontent.com/61646760/143177704-372d47ad-cb46-490e-b826-c12cbce3f069.png)
    ![image](https://user-images.githubusercontent.com/61646760/143177100-a45fc285-8077-4254-88a2-37f65c8f6d5e.png)
  - 의사 결정 나무 특징
    - 결과가 직관적이며, 해석하기 쉬움
    - 나무 깊이가 깊어질수록 과적합(Overfitting) 문제 발생 가능성이 매우 높음
      - 의사 결정 나무가 너무 깊으면 과적합을 야기할 수 있으므로 깊이의 균형(trade-off)이 중요함
    - 학습이 끝난 트리의 작업 속도가 매우 빠름
    
#### 간단한 의사 결정 나무 만들기
분류 설명에서 언급한 항공 지연 데이터를 기반으로 간단한 의사 결정 나무를 구현해 보자.

- 항공 지연 데이터는 아래와 같다.  
  ![image](https://user-images.githubusercontent.com/61646760/143177938-106246b4-daf0-4471-8b4d-748ae3ccb662.png)

- 풍속에 따른 지연 여부를 알아내기 위하여 의사 결정 나무인 `binary_tree`의 기준값(`threshold`)을 변경해 가며 완벽하게 지연 여부를 분리할 수 있는 모델을 만들어 보자.  
  ![image](https://user-images.githubusercontent.com/61646760/143177997-b865f09e-1f33-4ffe-8d5e-0528f7643001.png)

```
import numpy as np
import pandas as pd

# 풍속을 threshold 값에 따라 분리하는 의사결정나무 모델 함수
def binary_tree(data, threshold):
    
    yes = []
    no = []
    
    # data로부터 풍속 값마다 비교를 하기 위한 반복문
    for wind in data['풍속']:
    
        # threshold 값과 비교하여 분리합니다.
        if wind > threshold:
            yes.append(wind)
        else:
            no.append(wind)
    
    # 예측한 결과를 DataFrame 형태로 저장합니다.
    data_yes = pd.DataFrame({'풍속': yes, '예상 지연 여부': ['Yes']*len(yes)})  # yes의 길이만큼 지연 여부에 'Yes'를 넣어줌
    data_no = pd.DataFrame({'풍속': no, '예상 지연 여부': ['No']*len(no)})      # no의 길이만큼 지연 여 부에 'No'를 넣어줌
    
    return data_no.append(data_yes, ignore_index=True)  # 열 이름(column name) 무시하고 정수 번호 자동 부여

# 풍속에 따른 항공 지연 여부 데이터
Wind = [1, 1.5, 2.5, 5, 5.5, 6.5]
Delay  = ['No', 'No', 'No', 'Yes', 'Yes', 'Yes']

# 위 데이터를 DataFrame 형태로 저장합니다.
data = pd.DataFrame({'풍속': Wind, '지연 여부': Delay})
print(data,'\n')

'''
    풍속 지연 여부
0  1.0    No
1  1.5    No
2  2.5    No
3  5.0   Yes
4  5.5   Yes
5  6.5   Yes 
'''

"""
1. binary_tree 모델을 사용하여 항공 지연 여부를 예측합니다.
   (threshold에 값을 넣어서 결과를 확인)
"""
# 지연 여부 == 예상 지연 여부가 되도록 data_pred에 할당
data_pred = binary_tree(data, threshold = 3)  # 3으로 설정해야 data = data_pred가 됨
print(data_pred,'\n')

'''
    풍속 예상 지연 여부
0  1.0       No
1  1.5       No
2  2.5       No
3  5.0      Yes
4  5.5      Yes
5  6.5      Yes 
'''
```
- `binary_tree` 함수는 입력하는 `threshold` 풍속을 기준으로 지연 여부를 예측한 결과를 DataFrame 형태로 출력
- `threshold`에 값을 넣어서 결과를 확인 `예) 1, 2, 3.5, …`

#### 의사 결정 나무 - 불순도
- 의사 결정 나무 분리 기준
  - 데이터의 <strong>불순도(Impurity)</strong>를 최소화하는 구역으로 나누자!
- **불순도(Impurity)**
  - 다른 데이터가 섞여 있는 정도  
    ![image](https://user-images.githubusercontent.com/61646760/143276169-dbf138fd-c081-450f-9cd1-db3ba19774cb.png)
    - A가 불순도가 더 낮다.
    - 데이터의 개수가 적기 때문에 눈으로 확인 가능
    - 그렇다면 데이터 개수가 엄청 많을 때 불순도 측정은? 지니 불순도!
- **지니 계수(Gini Index)**
  - 해당 구역 안에서 특정 클래스에 속하는 데이터의 비율을 모두 제외한 값
    - 즉, **다양성**을 계산하는 방법

<p align="center">
 <image src="https://user-images.githubusercontent.com/61646760/143277757-abae407c-0f22-40c5-b710-bacd438bf20a.png" />
</p>

- **지니 불순도(Gini Impurity)**  
  - [지니 불순도 계산 과정](https://user-images.githubusercontent.com/61646760/143280573-8461ba58-a905-4a97-b744-352b66b28942.png)
    - [가장 낮은 지니 불순도 선택](https://user-images.githubusercontent.com/61646760/143281064-b13eded6-afaf-4edd-9bd9-4d24ab9d4c9d.png)
  - ![image](https://user-images.githubusercontent.com/61646760/143278477-2ada2df1-bfbd-4c72-915a-053669fd9014.png) : i번째 자식 마디의 데이터 개수
  - ![image](https://user-images.githubusercontent.com/61646760/143278540-b07146a0-835b-4026-b6f1-e73fbacc0c5f.png) : 부모 마디의 데이터 개수

<p align="center">
 <image src="https://user-images.githubusercontent.com/61646760/143278422-08459a10-968b-485b-abb0-165366c4d087.png" />  
</p>

#### sklearn을 사용한 의사결정나무 - 데이터 전 처리
Iris 데이터는 아래와 같이 꽃받침 길이, 꽃받침 넓이, 꽃잎 길이, 꽃잎 넓이의 네 가지 변수와 세 종류의 붓꽃 클래스로 구성되어 있다.

<p align="center">
 <image src="https://user-images.githubusercontent.com/61646760/143429627-1f1f43dc-d1bf-436a-a6da-5f35129ee133.png" />
</p>

**꽃받침 길이, 꽃받침 넓이, 꽃잎 길이, 꽃잎 넓이** 네 가지 변수가 주어졌을 때, 어떠한 붓꽃 종류인지 예측하는 분류 모델을 구현해 보자.

우선 데이터를 전 처리하기 위해서 4개의 변수를 갖는 feature 데이터와 `클래스` 변수를 label 데이터로 분리하고 학습용, 평가용 데이터로 나눠 보자.

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from elice_utils import EliceUtils
# elice_utils = EliceUtils()


# sklearn에 저장된 데이터를 불러옵니다.
X, Y = load_iris(return_X_y = True)

# DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']

"""
1. 학습용 평가용 데이터로 분리합니다
"""
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

# 원본 데이터 출력
print('원본 데이터 : \n',df.head(),'\n')

'''
원본 데이터 : 
    꽃받침 길이  꽃받침 넓이  꽃잎 길이  꽃잎 넓이  클래스
0     5.1     3.5    1.4    0.2    0
1     4.9     3.0    1.4    0.2    0
2     4.7     3.2    1.3    0.2    0
3     4.6     3.1    1.5    0.2    0
4     5.0     3.6    1.4    0.2    0 
'''

# 전 처리한 데이터 5개만 출력합니다
print('train_X : ')
print(train_X[:5],'\n')

'''
train_X : 
    꽃받침 길이  꽃받침 넓이  꽃잎 길이  꽃잎 넓이
22     4.6     3.6    1.0    0.2
15     5.7     4.4    1.5    0.4
65     6.7     3.1    4.4    1.4
11     4.8     3.4    1.6    0.2
42     4.4     3.2    1.3    0.2 
'''

print('train_Y : ')
print(train_Y[:5],'\n')

'''
train_Y : 
22    0
15    0
65    1
11    0
42    0
Name: 클래스, dtype: int64 
'''

print('test_X : ')
print(test_X[:5],'\n')

'''
test_X : 
     꽃받침 길이  꽃받침 넓이  꽃잎 길이  꽃잎 넓이
73      6.1     2.8    4.7    1.2
18      5.7     3.8    1.7    0.3
118     7.7     2.6    6.9    2.3
78      6.0     2.9    4.5    1.5
76      6.8     2.8    4.8    1.4 
'''

print('test_Y : ')
print(test_Y[:5])

'''
test_Y : 
73     1
18     0
118    2
78     1
76     1
Name: 클래스, dtype: int64
'''
```
- `load_iris`로 읽어 온 데이터 `X`에서 `Y`를 바탕으로 `train_test_split`을 사용하여 학습용:평가용 = 8:2 비율로 분리 (`random_state=42` 고정)

#### sklearn을 사용한 의사결정나무 - 학습하기
위에서 전 처리한 데이터를 바탕으로 의사결정나무 모델을 학습해 보자.

각 노드에서 불순도를 최소로 하는 의사결정나무 모델을 구현하기 위해서는 sklearn의 `DecisionTreeClassifier`을 사용한다.

- `DecisionTreeClassifier`
  - sklearn의 결정 트리 분류기
  - 먼저 해당 모델 객체를 불러와 초기화한다.
    - `DTmodel = DecisionTreeClassifier()`
    - 초기화를 수행할 때 `max_depth`를 설정하여 의사결정나무의 최대 깊이를 조절할 수 있다.
      - `DTmodel = DecisionTreeClassifier(max_depth=2)`
  - 초기화를 수행했다면 `fit` 함수와 전 처리된 데이터를 사용하여 학습을 수행한다.
    - `DTmodel.fit(train_X, train_Y)`

```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# from elice_utils import EliceUtils
# elice_utils = EliceUtils()


# sklearn에 저장된 데이터를 불러옵니다.
X, Y = load_iris(return_X_y = True)

# DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']
    
# 학습용 평가용 데이터로 분리합니다
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

# DTmodel에 의사결정나무 모델을 초기화 하고 학습합니다
DTmodel = DecisionTreeClassifier()
DTmodel.fit(train_X, train_Y)


# 학습한 결과를 출력합니다
plt.rc('font', family='NanumBarunGothic')
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DTmodel, 
                   feature_names=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'],  
                   class_names=['setosa', 'versicolor', 'virginica'],
                   filled=True)

fig.savefig("decision_tree.png")
# elice_utils.send_image("decision_tree.png")
```

![image](https://user-images.githubusercontent.com/61646760/143437769-216ecc5f-8517-483c-9503-27805c48bfad.png)

- sklearn의 `DecisionTreeClassifier()` 모델을 `DTmodel`에 초기화
- `fit`을 사용하여 `train_X`, `train_Y` 데이터를 학습

#### sklearn을 사용한 의사결정나무 - 예측하기
위에서 학습한 모델을 바탕으로 새로운 데이터에 대해서 예측해 보자.

`test_X` 데이터에 따른 예측값을 구해 보자.

- `predict()`
  - `DecisionTreeClassifier`를 사용하여 예측하는 함수
  - `pred_X = DTmodel.predict(test_X)`

```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# from elice_utils import EliceUtils
# elice_utils = EliceUtils()


# sklearn에 저장된 데이터를 불러옵니다.
X, Y = load_iris(return_X_y = True)

# DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']
    
# 학습용 평가용 데이터로 분리합니다
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

# DTmodel에 의사결정나무 모델을 초기화하고 학습합니다
DTmodel = DecisionTreeClassifier()
DTmodel.fit(train_X, train_Y)


# test_X에 대해서 예측합니다.
pred_X = DTmodel.predict(test_X)
print('test_X에 대한 예측값 : \n{}'.format(pred_X))

'''
test_X에 대한 예측값 : 
[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
'''
```

- `DTmodel`을 학습하고 `test_X`의 예측값을 구하여 `pred_X`에 저장


### 분류 평가 지표
#### 혼동 행렬
- **혼동 행렬(Confusion Matrix)**
  - 분류 모델의 성능을 평가하기 위함  
    ![image](https://user-images.githubusercontent.com/61646760/143467232-f0e1b4f3-548b-47c8-b9e2-595c37fcb833.png)
    - **True Positive** : 실제 Positive인 값을 Positive라고 예측(정답)
    - **True Negative** : 실제 Negative인 값을 Negative라고 예측(정답)
    - **False Positive** : 실제 Negative인 값을 Positive라고 예측(오답) - **1형 오류**
    - **False Negative** : 실제 Positive인 값을 Negative라고 예측(오답) - **2형 오류**

#### 정확도(Accuracy)
- **정확도(Accuracy)**  
  - 전체 데이터 중에서 제대로 분류된 데이터의 비율로, **모델이 얼마나 정확하게 분류하는지**를 나타냄  
  - 일반적으로 분류 모델의 주요 평가 방법으로 사용됨
  - 그러나, 클래스 비율이 불균형할 경우 평가 지표의 신뢰성을 잃을 가능성이 있음

<p align="center">
 <img src="https://user-images.githubusercontent.com/61646760/143470289-3d3dea7d-35a6-4805-b920-265f876c7300.png">
</p>

#### 정밀도(Precision)
- **정밀도(Precision)**
  - 모델이 Positive라고 분류한 데이터 중에서 실제로 Positive인 데이터의 비율
  - **Negative가 중요한 경우**
    - 즉, 실제로 Negative인 데이터를 Positive라고 판단하면 안 되는 경우 사용되는 지표
    - `예) 스팸 메일 판결을 위한 분류 문제`
      - 해당 메일이 스팸일 경우 Positive, 스팸이 아닐 경우 즉, 일반 메일일 경우 Negative
      - 일반 메일을 스팸 메일(Positive)로 잘못 예측했을 경우 중요한 메일을 전달받지 못하는 상황이 발생할 수 있음 

<p align="center">
 <img src="https://user-images.githubusercontent.com/61646760/143486879-aa177d38-93d0-4aa3-8407-bac19211af6e.png">
</p>

#### 재현율(Recall, TPR)
- **재현율(Recall, TPR)**
  - 실제로 Positive인 데이터 중에서 모델이 Positive로 분류한 데이터의 비율
  - **Positive가 중요한 경우**
    - 즉, 실제로 Positive인 데이터를 Negative라고 판단하면 안 되는 경우 사용되는 지표
    - `예) 악성 종양 여부 판결을 위한 검사`
      - 악성 종양일 경우 Positive, 악성 종양이 아닐 경우 즉, 양성 종양일 경우 Negative
      - 악성 종양(Positive)을 양성 종양(Negative)으로 잘못 예측했을 경우 제때 치료를 받지 못하게 되어 생명이 위급해질 수 있음

<p align="center">
 <img src="https://user-images.githubusercontent.com/61646760/143487539-af49a3ed-9981-442e-83eb-5bc9d5148fa3.png">
</p>
