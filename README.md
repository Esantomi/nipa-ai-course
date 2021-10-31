# nipa

### [2021 NIPA(National IT Industry Promotion Agency) AI online course](https://2021nipa.elice.io/explore)

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/135088173-e374261d-d1b7-40dc-a108-0e61d64df82f.png" width="35%" height="35%">
</p>

- [AI 실무 기본 과정](https://2021nipa.elice.io/tracks/1328/info)
  - [기본교육과정] 핵심 파이썬 기초 프로그래밍
  - [기본교육과정] 데이터 분석을 위한 라이브러리
  - [기본프로젝트] 공공 데이터를 활용한 파이썬 데이터 분석 프로젝트
- [AI 실무 응용 과정](https://2021nipa.elice.io/tracks/1329/info)
  - [응용교육과정] 머신러닝 시작하기
  - [응용교육과정] 딥러닝 시작하기
  - [응용교육과정] 산업별 AI혁신 사례 살펴보기
  - [응용프로젝트] 산업 데이터를 활용한 인공지능 프로젝트


## 교육 내용

### Python

- **매개변수(parameter)** : 함수를 정의할 때(만들 때) 넘겨받은 값을 관리하는 변수
- **인자(argument)** : 함수를 호출할 때(사용할 때) 함수로 넘겨주는 자료

#### 웹페이지 방문
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


### Numpy
Numpy 라이브러리는 효율적인 데이터 분석이 가능하도록 N차원의 배열 객체를 지원합니다.

Numpy의 배열은 파이썬의 list()보다도 빠른 연산과 효율적인 메모리 사용이 가능하기 때문에 빅데이터 분석 등에 널리 쓰이는 매우 강력한 라이브러리라고 할 수 있습니다.

#### 배열의 기초 : 1차원 배열
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

#### 배열의 기초 : 2차원 배열
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

#### Indexing & Slicing
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


### Pandas

Pandas는 구조화된 데이터를 효과적으로 처리하고 저장할 수 있는 Python 라이브러리입니다.

대용량 데이터를 쉽게 처리할 수 있는 Numpy를 기반으로 설계되어 있으며 Array 계산에 특화되어 있습니다.

#### Series
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

#### DataFrame
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

#### 데이터 선택 및 변경
- `.loc` : 명시적인 인덱스를 참조하는 인덱싱/슬라이싱
- `.iloc` : 정수 인덱스를 참조하는 인덱싱/슬라이싱
- `.query` : 조건에 부합하는 데이터를 추출할 때 사용하는 함수
  - query 함수 예시 : `country.query("population > 100000")`
  - masking 연산 예시 : `country[country['population'] < 10000]`

##### masking & query
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

##### 새로운 column 추가하기
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

#### 데이터프레임 정렬하기
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

#### 데이터프레임 분석용 함수
- `.count` : 데이터 개수 확인 (집계 함수, default: NaN 값 제외)
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

## 학습 계획
- ~2021-10-31 완료
