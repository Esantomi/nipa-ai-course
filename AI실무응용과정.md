# AI 실무 응용 과정

실제 데이터와 코드를 다뤄보며 최신 머신러닝/딥러닝 프레임워크와 라이브러리를 배워요!
- [2021 NIPA AI 온라인 교육 바로 가기](https://2021nipa.elice.io/tracks/1329/info)

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
