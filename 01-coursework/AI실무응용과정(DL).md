# AI 실무 응용 과정 - 딥러닝 시작하기

실제 데이터와 코드를 다뤄보며 최신 머신러닝/딥러닝 프레임워크와 라이브러리를 배워요!
- [2021 NIPA AI 온라인 교육 - AI 실무 응용 과정](https://2021nipa.elice.io/tracks/1329/info)

## 목차
- 딥러닝 시작하기
  - [01. 퍼셉트론](#01-퍼셉트론)
    - [딥러닝 개론](#딥러닝-개론)
    - [퍼셉트론(Perceptron)](#퍼셉트론Perceptron)
      - [퍼셉트론 작동 예시 구현하기](#퍼셉트론-작동-예시-구현하기)
      - [DIY 퍼셉트론 만들기](#DIY-퍼셉트론-만들기)
      - [퍼셉트론의 알맞은 가중치 찾기](#퍼셉트론의-알맞은-가중치-찾기)
    - [다층 퍼셉트론](#다층-퍼셉트론)
  - [02. 텐서플로우와 신경망](#02-텐서플로우와-신경망)
    - [딥러닝 모델의 학습 방법](#딥러닝-모델의-학습-방법)
      - [딥러닝 모델의 학습 순서 (정리)](#딥러닝-모델의-학습-순서)
    - [텐서플로우로 딥러닝 구현하기 - 데이터 전 처리](#텐서플로우로-딥러닝-구현하기---데이터-전-처리)
      - [딥러닝 모델 구현 순서](#딥러닝-모델-구현-순서)
      - [1. 데이터 전 처리하기](#1-데이터-전-처리하기)
      - [텐서플로우를 활용하여 신경망 구현하기 - 데이터 전 처리 실습](#텐서플로우를-활용하여-신경망-구현하기---데이터-전-처리)
    - [텐서플로우로 딥러닝 구현하기 - 모델 구현](#텐서플로우로-딥러닝-구현하기---모델-구현)
  - [03. 다양한 신경망](#03-다양한-신경망)

# 딥러닝 시작하기
- 수강 목표
  - 딥러닝에 대한 전반적인 이해
    - 딥러닝을 처음 접하는 사람에게 퍼셉트론부터 CNN, RNN까지 딥러닝에 대한 전반적인 내용을 학습
  - 이미지 및 자연어 처리를 위한 딥러닝 모델 학습
    - 이미지 및 자연어 데이터를 다루기 위한 전 처리 방식과 딥러닝 모델을 학습
  - 파이썬 기반 딥러닝 코딩 능력
    - 이론적인 딥러닝뿐만 아니라 코딩으로 수행할 수 있는 능력을 함양

## 01. 퍼셉트론
### 딥러닝 개론
- **딥러닝(Deep Learning)**
  - 머신러닝의 여러 방법론 중 하나
  - 인공신경망에 기반하여 컴퓨터에게 사람의 사고방식을 가르치는 방법
- **인공신경망(ANN, Artificial Neural Network)**
  - 생물학의 신경망에서 영감을 얻은 학습 알고리즘
  - 사람의 신경 시스템을 모방함
- 딥러닝의 역사
  ![image](https://user-images.githubusercontent.com/61646760/143669699-0e2ec31a-1e6b-42a1-9be5-7c9c8a5a0dd3.png)

### 퍼셉트론(Perceptron)
- **퍼셉트론(Perceptron)**
  - 프랑크 로젠블라트(Frank Rosenblatt)가 1957년에 제안한 초기 형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘
  - 각 노드의 가중치와 입력치를 곱한 것을 모두 합한 값이 <strong>활성화 함수(activation function)</strong>에 의해 판단되는데, 그 값이 임계치(보통 0)보다 크면 뉴런이 활성화되고 결과값으로 1을 출력한다. 뉴런이 활성화되지 않으면 결과값으로 -1을 출력한다.
    - [퍼셉트론 동작 예시](https://user-images.githubusercontent.com/61646760/143669359-7ae38bf5-fd08-4fb4-af2a-4f5c15e9862f.png)
  - 마빈 민스키와 시모어 페퍼트는 저서 "퍼셉트론"에서 단층 퍼셉트론은 XOR 연산이 불가능하지만, **다층 퍼셉트론으로는 XOR 연산이 가능함**을 보였다.
- 퍼셉트론의 기본 구조
  ![image](https://user-images.githubusercontent.com/61646760/143669215-0c6477c4-2aca-4150-b428-f65dfaa8738c.png)
  - [활성화 함수 그래프](https://user-images.githubusercontent.com/61646760/143669307-7a29d20a-9dc6-4fa4-baf0-bf3e79885786.png)
- 퍼셉트론을 활용한 선형 분류기
  - 퍼셉트론은 선형 분류기로써 데이터 분류 가능함
    - [예시 그래프](https://user-images.githubusercontent.com/61646760/143669470-d93987db-8dc4-4bcf-9a8b-c1cfba0aca79.png)
  - 하나의 선으로 분류할 수 없는 문제의 등장
    - 1969년 첫 번째 AI 겨울

#### 퍼셉트론 작동 예시 구현하기
이번 실습에서는 학습한 퍼셉트론의 작동 예시를 직접 수행해 보자.

- 신작 드라마 수(![image](https://user-images.githubusercontent.com/61646760/143670340-dca44f9e-17e6-4d1f-b926-69e24e431c35.png))와 확보한 여가 시간(![image](https://user-images.githubusercontent.com/61646760/143670350-d49d932f-3133-4b53-b004-df002a22c763.png))에 따른 엘리스 코딩 학습 여부(![image](https://user-images.githubusercontent.com/61646760/143670362-6c93eb80-836e-47ae-9388-ca92b970d587.png))를 예측하는 모델을 구현했을 때, 아래와 같은 변수들을 가정한다.

  - 변수 설명  
    ![image](https://user-images.githubusercontent.com/61646760/143670403-470e059d-f759-46ab-884e-4dfded7792ce.png)

- 아래 그래프는 ![image](https://user-images.githubusercontent.com/61646760/143670437-e53cf659-1f79-437f-9758-3025311cd0a7.png) 값을 갖는 Perceptron 함수의 결과를 출력한 것이다. 학습 여부(**출력값** ![image](https://user-images.githubusercontent.com/61646760/143670362-6c93eb80-836e-47ae-9388-ca92b970d587.png))가 1이 나오도록 ![image](https://user-images.githubusercontent.com/61646760/143670483-35b54a97-d460-46c6-805d-2410bf8b0e61.png) 값을 입력하자.  
  ![image](https://user-images.githubusercontent.com/61646760/143670502-d6919200-0e13-4ac3-9810-35434f513e4c.png)

```
# 학습 여부를 예측하는 퍼셉트론 함수
def Perceptron(x_1,x_2):
    
    # 설정한 가중치값을 적용
    w_0 = -5 
    w_1 = -1
    w_2 = 5
    
    # 활성화 함수에 들어갈 값을 계산
    output = w_0 + w_1 * x_1 + w_2 * x_2
    
    # 활성화 함수 결과를 계산
    if output < 0:
        y = 0
    else:
        y = 1
    
    return y, output


"""
1. perceptron의 예측 결과가 '학습한다:1'이 나오도록
   x_1, x_2에 적절한 값을 입력하세요. (상단 그래프 참고)
"""
x_1 = 2
x_2 = 4

result, go_out = Perceptron(x_1,x_2)

print("신호의 총합 : %d" % go_out)  # 신호의 총합 : 13

if go_out > 0:
    print("학습 여부 : %d\n ==> 학습한다!" % result)
else:
    print("학습 여부 : %d\n ==> 학습하지 않는다!" % result)

'''
학습 여부 : 1
 ==> 학습한다!
'''
```
- perceptron의 예측 결과가 '학습한다:1'이 나오도록 ![image](https://user-images.githubusercontent.com/61646760/143670483-35b54a97-d460-46c6-805d-2410bf8b0e61.png)에 적절한 값을 입력
  - 활성화 함수는 '**신호의 총합이 0 이상이면 학습하고, 0 미만이라면 학습하지 않는다**'는 규칙을 가진다.

#### DIY 퍼셉트론 만들기
![image](https://user-images.githubusercontent.com/61646760/143774048-8e6fefd3-83ea-47bf-9ed1-b870200de201.png)

이번 실습에선 위 그림과 같은 퍼셉트론을 구현해 보자.
- 위 그림에서 **m = 4**로 설정하겠다. 따라서 <strong>입력값(Input)</strong>은 ![image](https://user-images.githubusercontent.com/61646760/143774121-57ff7d74-4b01-44fa-9c4d-e95a4967a4ba.png)로 총 4개, <strong>가중치(Weight)</strong>도 ![image](https://user-images.githubusercontent.com/61646760/143774161-af35ed76-6f4a-46d8-afa7-002fe55706eb.png)로 총 4개이다.
- 가중치 ![image](https://user-images.githubusercontent.com/61646760/143774230-4d234b85-3520-4547-95d4-b0f9d41e9018.png)에 대한 입력값은 1이므로 두 값이 곱해진 값은 **상수** ![image](https://user-images.githubusercontent.com/61646760/143774230-4d234b85-3520-4547-95d4-b0f9d41e9018.png)이고, 이는 곧 **Bias**이다.
- 입력값 ![image](https://user-images.githubusercontent.com/61646760/143774121-57ff7d74-4b01-44fa-9c4d-e95a4967a4ba.png)와 가중치 ![image](https://user-images.githubusercontent.com/61646760/143774161-af35ed76-6f4a-46d8-afa7-002fe55706eb.png)까지 입력되면 ![image](https://user-images.githubusercontent.com/61646760/143774340-73e6ca34-2922-4cec-999b-f430c5054029.png), 즉 **신호의 총합** 값이 나오게 된다.
- ![image](https://user-images.githubusercontent.com/61646760/143774367-0865f5cc-c32d-4d37-a2e8-544bc4b3accb.png)은 이제 Activation function, 즉 활성화 함수에 입력값으로 들어가게 되고, 퍼셉트론은 ![image](https://user-images.githubusercontent.com/61646760/143774382-362f813b-f0bc-41f4-98c7-40e15386a895.png) 값으로 0 또는 1을 반환하게 된다.

```
'''
1. 신호의 총합과 그에 따른 결과 0 또는 1을
   반환하는 함수 perceptron을 완성합니다.
   
   Step01. 입력받은 값을 이용하여
           신호의 총합을 구합니다.
           
   Step02. 신호의 총합이 0 이상이면 1을, 
           그렇지 않으면 0을 반환하는
           활성화 함수를 작성합니다.
'''

def perceptron(w, x):
    
    output = w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]*x[3]
    
    if output >= 0:
        y = 1
    else:
        y = 0
    
    return y, output

# x_1, x_2, x_3, x_4의 값을 순서대로 list 형태로 저장
x = [1,2,3,4]

# w_0, w_1, w_2, w_3, w_4의 값을 순서대로 list 형태로 저장
w = [2, -1, 1, 3, -2]

# 퍼셉트론의 결과를 출력
y, output = perceptron(w,x)

print('output:', output)  # output: 4
print('y:', y)            # y: 1
```
- 신호의 총합 `output`을 정의하고, `output`이 0 이상이면 1을, 그렇지 않으면 0인 `y`를 반환하는 활성화 함수를 작성해 `perceptron` 함수를 완성

#### 퍼셉트론의 알맞은 가중치 찾기
이번 실습에서는 주어진 데이터를 완벽히 분리하는 퍼셉트론을 직접 구현해 보자.
- 단층 퍼셉트론을 직접 구현해 보며 적절한 **가중치(Weight), Bias** 값을 찾아보자.  
  ![image](https://user-images.githubusercontent.com/61646760/143895227-fe6f11fb-42b9-4a2e-8784-8151d9254367.png)

```
import numpy as np


def perceptron(w, x):
    
    output = w[1] * x[0] + w[2] * x[1] + w[0]
    
    if output >= 0:
        y = 1
    else:
        y = 0
    
    return y


# Input 데이터
X = [[0,0], [0,1], [1,0], [1,1]]  # x1, x2

'''
1. perceptron 함수의 입력으로 들어갈 가중치 값을 입력해 주세요.
   순서대로 w_0, w_1, w_2에 해당됩니다.
'''
w = [-2, 1, 1]  # x1, x2가 0일 때 y가 0이려면 w_0이 음수여야 한다. (활성화 함수는 값이 0 이상이면 활성화됨)

# AND Gate를 만족하는지 출력하여 확인
print('perceptron 출력')

for x in X:
    print('Input: ',x[0], x[1], ', Output: ',perceptron(w, x))

'''
perceptron 출력
Input: 0 0 , Output: 0
Input: 0 1 , Output: 0
Input: 1 0 , Output: 0
Input: 1 1 , Output: 1
'''
```

- `perceptron` 함수의 입력으로 들어갈 가중치 값을 입력
  - `w` 리스트 안의 값들은 순서대로 ![image](https://user-images.githubusercontent.com/61646760/143897909-7f2a6281-15a7-420f-a6d7-f36e370b9373.png)에 해당됨

### 다층 퍼셉트론
- **다층 퍼셉트론(Multi-Layer Perceptron)**
  - 단층 퍼셉트론을 여러 개 쌓아 입력층과 출력층 사이에 은닉층(hidden layer)으로 추가한 것
    - 단층 퍼셉트론은 입력층과 출력층만 존재
    - **은닉층(hidden layer)**
      - 입력층과 출력층 사이의 모든 Layer
      - 은닉층이 많아지면, 깊은 신경망이라는 의미의 Deep Learning 단어 사용
  - 1986년 첫 번째 빙하기의 끝

## 02. 텐서플로우와 신경망

### 딥러닝 모델의 학습 방법
- 딥러닝 모델
  - 히든층이 많아진다면, 깊은 신경망이라는 의미의 Deep Learning이라는 단어를 사용
    ![image](https://user-images.githubusercontent.com/61646760/144074964-3dac63cc-b64c-4325-b2f0-c77724771706.png)
  - 딥러닝 모델의 구성 요소
    - 레이어(Layer) : 모델을 구성하는 층
    - 노드(Node)/유닛(Unit) : 각 층을 구성하는 요소
    - 가중치(Weight) : 노드 간의 연결 강도
      - [구성 요소 이미지](https://user-images.githubusercontent.com/61646760/144075290-b0902cce-51e6-48fd-a799-2ab181932a6c.png)
  - 딥러닝 모델의 학습 방법
    - 예측값과 실제값 간의 오차값을 최소화하기 위해 **오차값을 최소화하는 모델의 인자를 찾는 알고리즘**을 적용
    - **Loss function을 최소화하는 가중치**를 찾기 위해 **최적화 알고리즘**을 적용
- 딥러닝 모델이 예측값 구하는 방식
  - **순전파(Forward propagation)**
    ![image](https://user-images.githubusercontent.com/61646760/144076803-6a647f4e-eae2-4144-87f1-d2aaec91013e.png)
    - 입력값을 바탕으로 출력값을 계산하는 과정
      - [순전파 예시](https://user-images.githubusercontent.com/61646760/144076957-1db5e0a9-b527-4e94-93d7-d063aa19baf1.png)
    - 순전파를 사용하면 예측값과 실제값 간의 오차값을 구하여 Loss function을 구할 수 있음
      - 그렇다면 어떻게 최적화를 해야 할까? <strong>경사하강법(Gradient descent)</strong>을 사용
  - **경사 하강법(Gradient descent)**
    - 가중치를 Loss function 값이 작아지게 업데이트하는 방법
    - 가중치는 **Gradient 값**을 사용하여 업데이트를 수행함
    - Gradient 값은 각 가중치마다 정해지며, <strong>역전파(Backpropagation)</strong>를 통하여 구할 수 있음
  - **역전파(Backpropagation)**
    ![image](https://user-images.githubusercontent.com/61646760/144083591-6ce72c04-079e-40d7-8243-93e56a65255c.png)
    - Forward propagation의 반대 방향으로 이루어지는 과정
  - 가중치 업데이트 과정
    ![image](https://user-images.githubusercontent.com/61646760/144083781-9b309a09-ec26-4901-bf1a-9a42cfef5df4.png) 
    - 위 과정을 수행하여 가중치들을 업데이트할 수 있으며, 이를 반복하여 **Loss function을 제일 작게 만드는 가중치**를 구함

#### 딥러닝 모델의 학습 순서
  1. 학습용 feature 데이터를 입력하여 예측값 구하기 **(순전파)**
  2. 예측값과 실제값 사이의 오차 구하기 **(Loss 계산)**
  3. Loss를 줄일 수 있는 가중치 업데이트하기 **(역전파)**
  4. 1~3번 반복으로 Loss를 최소로 하는 가중치 얻기

### 텐서플로우로 딥러닝 구현하기 - 데이터 전 처리

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/144258907-95520e72-7b7e-49e3-9bed-2270de30235c.png">
</p>

- **텐서플로우(TensorFlow)**  
  - 유연하고, 효율적이며, 확장성 있는 딥러닝 프레임워크
  - 대형 클러스터 컴퓨터부터 스마트폰까지 다양한 디바이스에서 동작 가능

#### 딥러닝 모델 구현 순서
1. 데이터 전 처리하기
2. 딥러닝 모델 구축하기
3. 모델 학습시키기
4. 평가 및 예측하기

#### 1. 데이터 전 처리하기
- Tensorflow 딥러닝 모델은 Tensor 형태의 데이터를 입력받는다.
  - **Tensor** : 다차원 배열로서 tensorflow에서 사용하는 객체
  - 따라서, 기존 데이터를 Tensor 형태로 변환한 뒤 모델에 사용
- `tf.data.Dataset`
  - Dataset API를 사용하여 딥러닝 모델용 Dataset을 생성
    ```
    # pandas를 사용하여 데이터 불러오기
    df=pd.read_csv('data.csv')
    feature = df.drop(columns=['label'])
    label = df['label']
    
    # tensor 형태로 데이터 변환
    dataset= tf.data.Dataset.from_tensor_slices((feature.values, label.values))
    ```
- Epoch와 Batch
  - 딥러닝에 사용하는 데이터는 추가적인 전 처리 작업이 필요함 => **Epoch, Batch**  
    ![image](https://user-images.githubusercontent.com/61646760/144260545-06bb13ac-3151-402e-ac6c-b6f2f5af0351.png)  
    - **Epoch** : 한 번의 epoch는 전체 데이터 셋에 대해 **한 번 학습을 완료한 상태**
    - **Batch** : 나눠진 데이터 셋 (보통 mini-batch라고 표현)
      - **iteration** : epoch를 나누어서 실행하는 횟수를 의미
  - `예) 총 데이터가 1000개, Batch size = 100`
    - 1 iteration = 100개 데이터에 대해서 학습
    - 1 epoch = 100 Batch size = 10 iteration
  - `tf.data.Dataset`
    ```
    # tensor 형태로 데이터 변환
    dataset = tf.data.Dataset.from_tensor_slices((feature.values, label.values))
    
    # dataset의 batch 사이즈를 32로 설정
    dataset = dataset.batch(32)
    ```

#### 텐서플로우를 활용하여 신경망 구현하기 - 데이터 전 처리
이번 실습에서는 텐서플로우를 활용하여 신경망을 구현해 보는 과정을 수행해 보자.
- 마케터로서 광고 비용에 따른 수익률을 신경망을 통해서 예측하고자 한다.
  - 아래와 같이 `FB`, `TV`, `Newspaper` 광고에 대한 비용 대비 `Sales` 데이터가 주어진다면 먼저 데이터 전 처리를 수행하여 텐서플로우 딥러닝 모델에 필요한 학습용 데이터를 만들면 된다.  
  ![image](https://user-images.githubusercontent.com/61646760/144263105-b8fda1a4-36d6-4b33-ab1f-071f80bcf9df.png)
- 텐서플로우 신경망 모델의 학습 데이터를 만드는 함수
  - 텐서플로우 신경망 모델의 학습 데이터는 기존 데이터를 `tf.data.Dataset` 형식으로 변환하여 사용한다.
  - **`from_tensor_slices()`**
    - pandas의 DataFrame 형태 데이터를 Dataset으로 변환하기 위해 사용하는 메서드
    - `ds = tf.data.Dataset.from_tensor_slices((X.values, Y.values))`
      - `X`는 feature 데이터가 저장된 DataFrame이고, `Y`는 label 데이터가 저장된 Series이다.
      - 여기서 `X`, `Y` 데이터는 `X.values`, `Y.values`를 사용하여 리스트 형태로 입력한다.
  - **`batch()`**
    - 이후 변환된 Dataset인 `ds`에 batch를 적용하고 싶다면 아래와 같이 `batch()` 메서드를 사용한다.
    - `ds = ds.shuffle(len(X)).batch(batch_size=5)`
      - `shuffle` : 데이터를 셔플하는 메서드. 인자로는 데이터의 크기를 입력함
      - `batch` : `batch_size`에 batch 크기를 넣게 되면 해당 크기로 batch를 수행하는 메서드
  - **`take()`**
    - 이렇게 처리한 `ds`에서 `take()` 메서드를 사용하면 batch로 분리된 데이터를 확인할 수 있다.

```
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 환경 변수 설정

np.random.seed(100)
tf.random.set_seed(100)

# 데이터를 DataFrame 형태로 불러옵니다.
df = pd.read_csv("data/Advertising.csv")

# DataFrame 데이터 샘플 5개를 출력합니다.
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

# 의미 없는 변수는 삭제합니다.
df = df.drop(columns=['Unnamed: 0'])

"""
1. Sales 변수는 label 데이터로 Y에 저장하고 나머진 X에 저장합니다.
"""
X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

"""
2. 학습용 데이터를 tf.data.Dataset 형태로 변환합니다.
   from_tensor_slices 함수를 사용하여 변환하고 batch를 수행하게 합니다.
"""
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

# 하나의 batch를 뽑아서 feature와 label로 분리합니다.
[(train_features_batch, label_batch)] = train_ds.take(1)

# batch 데이터를 출력합니다.
print('FB, TV, Newspaper batch 데이터:\n', train_features_batch)
print('\nSales batch 데이터:', label_batch)

'''
FB, TV, Newspaper batch 데이터:
 tf.Tensor(
[[296.4  36.3 100.9]
 [228.   37.7  32. ]
 [  5.4  29.9   9.4]
 [ 57.5  32.8  23.5]
 [240.1   7.3   8.7]], shape=(5, 3), dtype=float64)
 
Sales batch 데이터: tf.Tensor([23.8 21.5  5.3 11.8 13.2], shape=(5,), dtype=float64)
'''
```

### 텐서플로우로 딥러닝 구현하기 - 모델 구현

## 03. 다양한 신경망
