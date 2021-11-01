# AI 실무 응용 과정

- [2021 NIPA AI 온라인 교육 바로 가기](https://2021nipa.elice.io/tracks/1329/info)

## 머신러닝 시작하기

### 01. 자료 형태의 이해
- **수치형 자료(Numerical data)** : 양적 자료(Quantitative data) `예) 키, 몸무게, 성적, 나이`
  - 연속형 자료(Continuous data) : 연속적인 관측값을 가짐 `예) 원주율(3.14159...), 시간`
  - 이산형 자료(Discrete data) : 셀 수 있는 관측값을 가짐 `예) 뉴스 글자 수, 주문 상품 개수`
- **범주형 자료(Categorical data)** : 질적 자료(Qualitative data) `예) 성별, 지역, 혈액형`
  - 순위형 자료(Ordinal data) : 범주 사이의 순서에 의미가 있음 `예) 학점 (A+, A, A-)`
  - 명목형 자료(Nominal data) : 범주 사이의 순서에 의미가 없음 `예) 혈액형 (A, B, O, AB)`
- 혼동되는 경우
  - 범주형 자료는 숫자로 표기해도 범주형 자료 (남자 1, 여자 0)
  - 수치형 자료는 구간화하면 범주형 자료 (10 ~ 19세, 20 ~ 29세)

### 02. 범주형 자료의 요약
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
