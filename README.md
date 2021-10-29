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

### Numpy
Numpy 라이브러리는 효율적인 데이터 분석이 가능하도록 N차원의 배열 객체를 지원합니다.

Numpy의 배열은 파이썬의 list()보다도 빠른 연산과 효율적인 메모리 사용이 가능하기 때문에 빅데이터 분석 등에 널리 쓰이는 매우 강력한 라이브러리라고 할 수 있습니다.

#### 배열의 기초
```
import numpy as np

array = np.array(range(10))  # 1차원 array
print(array)                 # [0 1 2 3 4 5 6 7 8 9]

# 1. array의 자료형을 출력해보세요.
print(type(array))  # <class 'numpy.ndarray'>

# 2. array의 차원을 출력해보세요.
print(array.ndim)   # 1

# 3. array의 모양을 출력해보세요.
print(array.shape)  # (10,)

# 4. array의 크기를 출력해보세요.
print(array.size)   # 10

# 5. array의 dtype(data type)을 출력해보세요.
print(array.dtype)  # int64

# 6. array의 인덱스 5의 요소를 출력해보세요.
print(array[5])     # 5

# 7. array의 인덱스 3의 요소부터 인덱스 5 요소까지 출력해보세요.
print(array[3:6])   # [3 4 5]
```

## 학습 계획
- ~2021-10-31 완료
