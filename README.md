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

#### Dataframe
시리즈 데이터는 하나의 컬럼 값으로 이루어진 반면 데이터 프레임은 여러 개의 컬럼 값을 가질 수 있습니다.

## 학습 계획
- ~2021-10-31 완료
