# AI 실무 기본 과정
파이썬 기초부터 데이터 분석까지, 실습을 통해 인공지능을 위한 기초 체력을 다져요!
- [2021 NIPA AI 온라인 교육 - AI 실무 기본 과정](https://2021nipa.elice.io/tracks/1328/info)

## 목차
- [Python](#python)
  * [웹페이지 방문](#웹페이지-방문)
- [Numpy](#numpy)
  * [배열의 기초 : 1차원 배열](#배열의-기초--1차원-배열)
  * [배열의 기초 : 2차원 배열](#배열의-기초--2차원-배열)
  * [Indexing & Slicing](#indexing--slicing)
- [Pandas](#pandas)
  * [Series](#series)
  * [DataFrame](#dataframe)
  * [데이터 선택 및 변경](#데이터-선택-및-변경)
    + [masking & query](#masking--query)
    + [새로운 column 추가하기](#새로운-column-추가하기)
  * [데이터프레임 정렬하기](#데이터프레임-정렬하기)
  * [데이터프레임 분석용 함수](#데이터프레임-분석용-함수)
  * [그룹으로 묶기](#그룹으로-묶기)
    + [groupby()](#groupby)
    + [aggregate()](#aggregate)
- [Matplotlib](#matplotlib)
  * [선 그래프](#선-그래프Line-Graph)
  * [그래프 범례](#그래프-범례)
  * [막대 그래프 & 히스토그램](#막대-그래프--히스토그램)
  * [Matplotlib with Pandas](#Matplotlib-with-Pandas)

## Python

- **매개변수(parameter)** : 함수를 정의할 때(만들 때) 넘겨받은 값을 관리하는 변수
- **인자(argument)** : 함수를 호출할 때(사용할 때) 함수로 넘겨주는 자료

### 웹페이지 방문
Python에서는 쉽게 웹페이지의 정보를 가져올 수 있는 urllib 패키지를 제공합니다.

이 중에서 `urllib.request.urlopen` 함수는 해당 url의 html 파일을 가져옵니다.

```
from urllib.request import urlopen

webpage = urlopen("https://en.wikipedia.org/wiki/Lorem_ipsum").read().decode("utf-8")

print(webpage)

'''
<!DOCTYPE html>
<html class="client-nojs" lang="en" dir="ltr">
...
'''
```

- `urlopen()` : 이 함수에 url을 넣으면 해당 url에 접근한 결과를 얻을 수 있습니다.
- `read()` : 결과를 해독하여 문자열로 반환합니다.
- `decode()` : 문자열을 특정 인코딩 방식으로 해독합니다. (유니코드, 아스키 코드 등)


## Numpy
Numpy 라이브러리는 효율적인 데이터 분석이 가능하도록 N차원의 배열 객체를 지원합니다.

Numpy의 배열은 파이썬의 list()보다도 빠른 연산과 효율적인 메모리 사용이 가능하기 때문에 빅데이터 분석 등에 널리 쓰이는 매우 강력한 라이브러리라고 할 수 있습니다.

### 배열의 기초 : 1차원 배열
```
import numpy as np

array = np.array(range(10))  # 1차원 array
print(array)                 # [0 1 2 3 4 5 6 7 8 9]

# 1. array의 자료형을 출력해 보세요.
print(type(array))  # <class 'numpy.ndarray'>

# 2. array의 차원을 출력해 보세요.
print(array.ndim)   # 1

# 3. array의 모양을 출력해 보세요.
print(array.shape)  # (10,)

# 4. array의 크기를 출력해 보세요.
print(array.size)   # 10

# 5. array의 dtype(data type)을 출력해 보세요.
print(array.dtype)  # int64

# 6. array의 인덱스 5의 요소를 출력해 보세요.
print(array[5])     # 5

# 7. array의 인덱스 3의 요소부터 인덱스 5 요소까지 출력해 보세요.
print(array[3:6])   # [3 4 5]
```

### 배열의 기초 : 2차원 배열
```
import numpy as np

matrix = np.array(range(1,16))  # 2차원 배열
matrix.shape = 3,5              # 1부터 15까지 들어있는 (3,5)짜리 배열을 만듭니다.
print(matrix)

'''
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]]
'''

# 1. matrix의 자료형을 출력해 보세요.
print(type(matrix))  # <class 'numpy.ndarray'>

# 2. matrix의 차원을 출력해 보세요.
print(matrix.ndim)   # 2

# 3. matrix의 모양을 출력해 보세요.
print(matrix.shape)  # (3, 5)

# 4. matrix의 크기를 출력해 보세요.
print(matrix.size)   # 15

# 5. matrix의 dtype(data type)을 출력해 보세요.
print(matrix.dtype)  # int64

# 6. matrix의 dtype을 str로 변경하여 출력해 보세요.
print(matrix.astype('str'))

'''
[['1' '2' '3' '4' '5']
 ['6' '7' '8' '9' '10']
 ['11' '12' '13' '14' '15']]
'''

# 7. matrix의 (2,3) 인덱스의 요소를 출력해 보세요.
print(matrix[2, 3])  # 14

# 8. matrix의 행은 인덱스 0부터 인덱스 1까지, 열은 인덱스 1부터 인덱스 3까지 출력해 보세요.
print(matrix[0:2, 1:4])

'''
[[2 3 4]
 [7 8 9]]
'''
```

### Indexing & Slicing
```
import numpy as np

matrix = np.arange(1, 13, 1).reshape(3, 4)
print(matrix)

'''
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
'''

# 1. Indexing을 통해 값 2를 출력해 보세요.
answer1 = matrix[0, 1]        # 1행 2열

# 2. Slicing을 통해 매트릭스 일부인 9, 10을 가져와 출력해 보세요.
answer2 = matrix[2:, :2]      # 2행부터 마지막 행까지, 1열부터 2열까지

# 3. Boolean indexing을 통해 5보다 작은 수를 찾아 출력해 보세요.
answer3 = matrix[matrix < 5]  # Boolean mask를 이용하여 원하는 값을 추출하는 방식

# 4. Fancy indexing을 통해 두 번째 행만 추출하여 출력해 보세요.
answer4 = matrix[[1]]         # 배열의 각 요소 선택을 Index 배열을 전달하여 지정하는 방식

# 위에서 구한 정답을 출력해 봅시다.
print(answer1)  # 2
print(answer2)  # [[ 9 10]]
print(answer3)  # [1 2 3 4]
print(answer4)  # [[5 6 7 8]]
```

- `answer2 = matrix[2, :2]`로 하거나 `answer4 = matrix[1]`로 하면 각각 `[ 9 10]`, `[5 6 7 8]`이 나온다. 2차원 배열임에 유의할 것!


## Pandas

Pandas는 구조화된 데이터를 효과적으로 처리하고 저장할 수 있는 Python 라이브러리입니다.

대용량 데이터를 쉽게 처리할 수 있는 Numpy를 기반으로 설계되어 있으며 Array 계산에 특화되어 있습니다.

### Series
Series 데이터란 Numpy array가 보강된 형태로, Data와 index를 가지고 있는 데이터 형식입니다.

```
import numpy as np
import pandas as pd

# 예시) 시리즈 데이터를 만드는 방법.
series = pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'], name="Title")
print(series, "\n")

'''
a    1
b    2
c    3
d    4
Name: Title, dtype: int64 
'''

# 국가별 인구 수 시리즈 데이터를 딕셔너리를 사용하여 만들어 보세요.
dict = {
'korea' : 5180,
'japan' : 12718,
'china' : 141500,
'usa' : 32676
}

country = pd.Series(dict)
print(country)

'''
korea      5180
japan     12718
china    141500
usa       32676
dtype: int64
'''
```

### DataFrame
DataFrame은 여러 개의 Series가 모여서 행과 열을 이룬 데이터를 말합니다.

따라서 시리즈 데이터는 하나의 컬럼 값으로 이루어진 반면, 데이터 프레임은 여러 개의 컬럼 값을 가질 수 있습니다.

```
import numpy as np
import pandas as pd

# 두 개의 시리즈 데이터가 있습니다.
print("Population series data:")
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)
print(population, "\n")

'''
Population series data:
korea      5180
japan     12718
china    141500
usa       32676
dtype: int64 
'''

print("GDP series data:")
gdp_dict = {
    'korea': 169320000,
    'japan': 516700000,
    'china': 1409250000,
    'usa': 2041280000,
}
gdp = pd.Series(gdp_dict)
print(gdp, "\n")

'''
GDP series data:
korea     169320000
japan     516700000
china    1409250000
usa      2041280000
dtype: int64 
'''

# 이곳에서 2개의 시리즈 값이 들어간 데이터프레임을 생성합니다.
print("Country DataFrame")
country = pd.DataFrame({
'population' : population,
'gdp' : gdp
})
print(country, "\n")

'''
Country DataFrame
       population         gdp
korea        5180   169320000
japan       12718   516700000
china      141500  1409250000
usa         32676  2041280000 
'''

# DataFrame의 index, column을 확인해 봅시다.
print(country.index)    # Index(['korea', 'japan', 'china', 'usa'], dtype='object')
print(country.columns)  # Index(['population', 'gdp'], dtype='object')
```

- Series가 DataFrame의 열(column)로 들어간다는 것을 확인할 수 있습니다.

### 데이터 선택 및 변경
- `.loc` : 명시적인 인덱스를 참조하는 인덱싱/슬라이싱
- `.iloc` : 정수 인덱스를 참조하는 인덱싱/슬라이싱
- `.query` : 조건에 부합하는 데이터를 추출할 때 사용하는 함수
  - query 함수 예시 : `country.query("population > 100000")`
  - masking 연산 예시 : `country[country['population'] < 10000]`

#### masking & query
```
import numpy as np
import pandas as pd

print("Masking & query")
df = pd.DataFrame(np.random.rand(5, 2), columns=["A", "B"])
print(df, "\n")

'''
Masking & query
          A         B
0  0.947358  0.927704
1  0.238985  0.417608
2  0.239408  0.178519
3  0.592118  0.637146
4  0.955686  0.474632 
'''

# 데이터 프레임에서 A 컬럼 값이 0.5보다 작고 B 컬럼 값이 0.3보다 큰 값들을 구해 봅시다.
# 마스킹 연산을 활용하여 출력해 보세요!
print(df[(df['A'] < 0.5) & (df['B'] > 0.3)])

'''
          A         B
1  0.238985  0.417608
'''

# query 함수를 활용하여 출력해보세요!
print(df.query("A < 0.5 and B > 0.3"))

'''
          A         B
1  0.238985  0.417608
'''
```

#### 새로운 column 추가하기
```
import numpy as np
import pandas as pd

# GDP와 인구수 시리즈 값이 들어간 데이터프레임을 생성합니다.
population = pd.Series({'korea': 5180,'japan': 12718,'china': 141500,'usa': 32676})
gdp = pd.Series({'korea': 169320000,'japan': 516700000,'china': 1409250000,'usa': 2041280000})

print("Country DataFrame")
country = pd.DataFrame({"population" : population,"gdp" : gdp})
print(country)

'''
Country DataFrame
       population         gdp
korea        5180   169320000
japan       12718   516700000
china      141500  1409250000
usa         32676  2041280000
'''

# 데이터프레임에 gdp per capita 칼럼을 추가하고 출력합니다.
gdp_per_capita = gdp / population
# gdp_per_capita = country["gdp"] / country["population"]
country["gdp per capita"] = gdp_per_capita
print(country)

'''
       population         gdp  gdp per capita
korea        5180   169320000    32687.258687
japan       12718   516700000    40627.457147
china      141500  1409250000     9959.363958
usa         32676  2041280000    62470.314604
'''
```

### 데이터프레임 정렬하기
- `.sort_values` : 값으로 정렬
- `.sort_index` : 인덱스로 정렬
  - `axis = 0` : 행 인덱스 기준 정렬 (default가 오름차순이므로 `ascending = True` 생략함)
  - `axis = 1, ascending = False` : 열 인덱스 기준 정렬 (내림차순)

```
import numpy as np
import pandas as pd

print("DataFrame: ")
df = pd.DataFrame({
    'col1' : [2, 1, 9, 8, 7, 4],
    'col2' : ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col3': [0, 1, 9, 4, 2, 3],
})
print(df, "\n")

'''
DataFrame: 
   col1 col2  col3
0     2    A     0
1     1    A     1
2     9    B     9
3     8  NaN     4
4     7    D     2
5     4    C     3 
'''


# 정렬 코드 입력해 보기    
# 1. col1을 기준으로 오름차순으로 정렬하기.
sorted_df1 = df.sort_values('col1', ascending = True)


# 2. col2를 기준으로 내림차순으로 정렬하기.
sorted_df2 = df.sort_values('col2', ascending = False)


# 3. col2를 기준으로 오름차순으로, col1를 기준으로 내림차순으로 정렬하기.
sorted_df3 = df.sort_values(['col2', 'col1'], ascending=[True, False])


# 출력
print("sorted_df1: ")
print(sorted_df1, "\n")

'''
sorted_df1: 
   col1 col2  col3
1     1    A     1
0     2    A     0
5     4    C     3
4     7    D     2
3     8  NaN     4
2     9    B     9 
'''

print("sorted_df2: ")
print(sorted_df2, "\n")

'''
sorted_df2: 
   col1 col2  col3
4     7    D     2
5     4    C     3
2     9    B     9
0     2    A     0
1     1    A     1
3     8  NaN     4 
'''

print("sorted_df3: ")
print(sorted_df3)

'''
sorted_df3: 
   col1 col2  col3
0     2    A     0
1     1    A     1
2     9    B     9
5     4    C     3
4     7    D     2
3     8  NaN     4
'''
```

### 데이터프레임 분석용 함수
**집계 함수**는 많은 데이터 값을 입력으로 받아 이를 요약하는 하나의 값을 반환하는 기능을 합니다.
- `.count` : 데이터 개수 확인 (default: NaN 값 제외)
  - `axis = 0` : 열 기준
  - `axis = 1` : 행 기준
- `.max` : 최댓값
- `.min` : 최솟값
- `.sum` : 합계
  - `axis = 1` : 행 기준
- `.mean` : 평균
  - `axis = 1` : 행 기준
  - `skipna = True` : NaN 생략
- `.fillna` : NaN 값을 다른 값으로 대체

```
import numpy as np
import pandas as pd

data = {
    'korean' : [50, 60, 70],
    'math' : [10, np.nan, 40]
}
df = pd.DataFrame(data, index = ['a','b','c'])
print(df, "\n")

'''
   korean  math
a      50  10.0
b      60   NaN
c      70  40.0 
'''

# 각 컬럼별 데이터 개수
col_num = df.count(axis = 0)
print(col_num, "\n")

'''
korean    3
math      2
dtype: int64 
'''

# 각 행별 데이터 개수
row_num = df.count(axis = 1)
print(row_num, "\n")

'''
a    2
b    1
c    2
dtype: int64 
'''

# 각 컬럼별 최댓값
col_max = df.max()
print(col_max, "\n")

'''
korean    70.0
math      40.0
dtype: float64 
'''

# 각 컬럼별 최솟값
col_min = df.min()
print(col_min, "\n")

'''
korean    50.0
math      10.0
dtype: float64 
'''

# 각 컬럼별 합계
col_sum = df.sum()
print(col_sum, "\n")

'''
korean    180.0
math       50.0
dtype: float64 
'''

# 컬럼의 최솟값으로 NaN값 대체
math_min = df['math'].min()
df['math'] = df['math'].fillna(math_min)
print(df, "\n")

'''
   korean  math
a      50  10.0
b      60  10.0
c      70  40.0 
'''

# 각 컬럼별 평균
col_avg = df.mean()
print(col_avg, "\n")

'''
korean    60.0
math      20.0
dtype: float64 
'''
```

### 그룹으로 묶기
- `.groupby` : key 값을 기준으로 group으로 묶기
  - `.aggregate` : 한 번의 groupby를 통해 여러 개의 통계 함수를 적용
  - `.filter` : groupby를 통해서 그룹 속성을 기준으로 데이터 필터링
  - `.apply` : groupby를 통해서 묶인 데이터에 함수 적용 (패러미터로 lambda 사용)
  - `.get_group` : groupby로 묶인 데이터에서 key 값으로 데이터를 가져옴

#### groupby()
```
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': [1, 2, 3, 1, 2, 3],
    'data2': [4, 4, 6, 0, 6, 1]
})
print("DataFrame:")
print(df, "\n")

'''
DataFrame:
  key  data1  data2
0   A      1      4
1   B      2      4
2   C      3      6
3   A      1      0
4   B      2      6
5   C      3      1 
'''

# groupby 함수를 이용해 봅시다.
# key를 기준으로 묶어 합계를 구해 출력해 보세요.
print(df.groupby('key').sum())

'''
     data1  data2
key              
A        2      4
B        4     10
C        6      7
'''

# key와 data1을 기준으로 묶어 합계를 구해 출력해 보세요.
print(df.groupby(['key', 'data1']).sum())

'''
           data2
key data1       
A   1          4
B   2         10
C   3          7
'''
```

#### aggregate()
```
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': [0, 1, 2, 3, 4, 5],
    'data2': [4, 4, 6, 0, 6, 1]
})
print("DataFrame:")
print(df, "\n")

'''
DataFrame:
  key  data1  data2
0   A      0      4
1   B      1      4
2   C      2      6
3   A      3      0
4   B      4      6
5   C      5      1 
'''

# aggregate를 이용하여 요약 통계량을 산출해 봅시다.
# 데이터 프레임을 'key' 칼럼으로 묶고, data1과 data2 각각의 최솟값, 중앙값, 최댓값을 출력하세요.
print(df.groupby('key').aggregate([min, np.median, max]), "\n")

'''
    data1            data2           
      min median max   min median max
key                                  
A       0    1.5   3     0    2.0   4
B       1    2.5   4     4    5.0   6
C       2    3.5   5     1    3.5   6 
'''


# 데이터 프레임을 'key' 칼럼으로 묶고, data1의 최솟값, data2의 합계를 출력하세요.
print(df.groupby('key').aggregate({'data1':min, 'data2':sum}))

'''
     data1  data2
key              
A        0      4
B        1     10
C        2      7
'''
```

## Matplotlib

### 선 그래프(Line Graph)
![image](https://user-images.githubusercontent.com/61646760/141137423-fc24ca96-5566-4583-89df-87767e0f07a5.png)

```
# from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# elice_utils = EliceUtils()

# 이미 입력되어 있는 코드의 다양한 속성 값들을 변경해 봅시다.
x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(
    x, x, label='y=x',
    linestyle='-',   # 실선
    marker='.',      # 점
    color='blue'     # 파란색
)
ax.plot(
    x, x**2, label='y=x^2',
    linestyle='-.',  # 대시 점선
    marker=',',      # 픽셀
    color='red'      # 빨간색
)

# elice에서 그래프를 확인
fig.savefig("plot.png")
# elice_utils.send_image("plot.png")
```

![image](https://user-images.githubusercontent.com/61646760/141138053-43942eb7-ce0c-475a-83ce-12dd25d79ef2.png)
- 상기 예시 코드에서 figure와 ax가 각각 1개인데, `ax.plot` 함수를 두 번 호출했다. 이렇게 하나의 ax에 여러 번 그래프를 그리면, 그래프를 겹쳐서 그릴 수 있다.

### 그래프 범례
- `ax.legend()` : 그래프의 데이터 정보, 즉 범례(legend)를 띄워 주는 메서드
  - 위치 옵션  
  ![image](https://user-images.githubusercontent.com/61646760/141326070-8e5383a5-8945-470b-98dc-d9acf9b0aaf8.png)

```
# from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# elice_utils = EliceUtils()

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(
    x, x, label='y=x',
    linestyle='-',
    marker='.',
    color='blue'
)
ax.plot(
    x, x**2, label='y=x^2',
    linestyle='-.',
    marker=',',
    color='red'
)
ax.set_xlabel("x")
ax.set_ylabel("y")

# 이미 입력되어 있는 코드의 다양한 속성값들을 변경해 봅시다.
ax.legend(
    loc='center left',  # 중간 왼쪽
    shadow=True,
    fancybox=True,
    borderpad=2
)

# elice에서 그래프를 확인
fig.savefig("plot.png")
# elice_utils.send_image("plot.png")
```
![image](https://user-images.githubusercontent.com/61646760/141326511-d0ddcbd7-a8c7-419d-9518-9cf0d1437027.png)

### 막대 그래프 & 히스토그램
- **막대 그래프(Bar chart)** : 여러 값을 비교하는 데 적합하다. 여러 개의 값을 입력받고 그 값들을 한눈에 비교할 수 있다.
- **히스토그램(Histogram)** : 일정 시간 동안의 숫자 데이터 분포를 시각화하는 데 적합하다.

```
# from elice_utils import EliceUtils
# elice_utils = EliceUtils()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fname='./NanumBarunGothic.ttf'
font = fm.FontProperties(fname = fname).get_name()
plt.rcParams["font.family"] = font

# Data set
x = np.array(["축구", "야구", "농구", "배드민턴", "탁구"])  # 스포츠 종목
y = np.array([13, 10, 17, 8, 7])                            # 선호하는 학생 수
z = np.random.randn(1000)                                   # 1,000개의 정규분포 난수

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Bar 그래프
axes[0].bar(x, y)

# 히스토그램
axes[1].hist(z, bins = 200)

# elice에서 그래프 확인하기
fig.savefig("plot.png")
# elice_utils.send_image("plot.png")
```

![image](https://user-images.githubusercontent.com/61646760/141612559-9a95f217-00cd-4b8f-9ceb-3ac0e5ed8371.png)

- 아래 코드는, 하나의 도화지(figure)에 `1*2`의 모양으로 그래프를 그리도록 합니다. 즉, 그래프를 2개 그리고, 가로로 배치한다는 의미입니다.  
  ```
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  ```
  - `axes[0]`은 막대 그래프를, `axes[1]`은 히스토그램을 그립니다.
- matplotlib의 pyplot으로 그래프를 그릴 때, 기본 폰트는 한글을 지원하지 않습니다. 아래는 한글을 지원하는 나눔바른고딕 폰트로 바꾼 코드입니다.
  ```
  import matplotlib.font_manager as fm
  
  fname='./NanumBarunGothic.ttf'
  font = fm.FontProperties(fname = fname).get_name()
  plt.rcParams["font.family"] = font
  ```
  - 위 코드 덕분에, 막대 그래프에서 `축구, 야구, 농구, 배드민턴, 탁구`가 올바르게 출력되었습니다.

### Matplotlib with Pandas
Pandas를 이용해 csv 파일을 불러와 시각화를 진행해 보도록 한다.

포켓몬에 대한 데이터가 담긴 csv 파일을 불러와서 공격 타입에 따라 다른 색상을 띠는 산점도 그래프를 그리고 라벨을 한번 붙여 보도록 하자.

공격 능력치와 수비 능력치가 x와 y축으로 주어지고, 물 타입 포켓몬은 파란색, 불 타입 포켓몬은 빨간색으로 표현하도록 하자.
