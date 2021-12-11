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
      - [2. 딥러닝 모델 구축하기](#2-딥러닝-모델-구축하기)
      - [3. 딥러닝 모델 학습시키기](#3-딥러닝-모델-학습시키기)
      - [4. 평가 및 예측하기](#4-평가-및-예측하기)
      - [텐서플로우를 활용하여 신경망 구현하기 - 모델 구현 실습](#텐서플로우를-활용하여-신경망-구현하기---모델-구현)
      - [텐서플로우를 활용하여 신경망 구현하기 - 모델 학습 실습](#텐서플로우를-활용하여-신경망-구현하기---모델-학습)
      - [텐서플로우를 활용하여 신경망 구현하기 - 모델 평가 및 예측 실습](#텐서플로우를-활용하여-신경망-구현하기---모델-평가-및-예측)
      - [신경망 모델로 분류하기 실습](#신경망-모델로-분류하기)
  - [03. 다양한 신경망](#03-다양한-신경망)
    - [이미지 처리를 위한 데이터 전 처리](#이미지-처리를-위한-데이터-전-처리)
      - [이미지 전 처리하기](#이미지-전-처리하기)
    - [MNIST 분류 CNN 모델 - 데이터 전 처리 실습](#MNIST-분류-CNN-모델---데이터-전-처리)
      - [CNN을 위한 데이터 전 처리](#CNN을-위한-데이터-전-처리)
    - [이미지 처리를 위한 딥러닝 모델](#이미지-처리를-위한-딥러닝-모델)
    - [MNIST 분류 CNN 모델 - 모델 구현 실습](#MNIST-분류-CNN-모델---모델-구현)
    - [MNIST 분류 CNN 모델 - 평가 및 예측 실습](#MNIST-분류-CNN-모델---평가-및-예측)

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

- pandas DataFrame `df`에서 `Sales` 변수는 label 데이터로 `Y`에 저장하고 나머진 `X`에 저장한다.
- 학습용 데이터 `train_X`, `train_Y`를 `tf.data.Dataset` 형태로 변환한다.
  - `from_tensor_slices` 함수를 사용하여 변환한다.

### 텐서플로우로 딥러닝 구현하기 - 모델 구현

<p align="center">
  <img src="https://user-images.githubusercontent.com/61646760/144272320-1508437c-30bd-4e1d-98e8-178fb8a0515c.png">
</p>

- **케라스(Keras)**
  - 텐서플로우의 패키지로 제공되는 고수준 API
  - 딥러닝 모델을 간단하고 빠르게 구현 가능
- 딥러닝 모델 구축을 위한 Keras 메소드
  - `tf.keras.models.Sequential()`
    - 모델 클래스 객체 생성
  - `tf.keras.layers.Dense(units, activation)`
    - 모델의 각 Layer 구성
    - `units` : 레이어 안의 Node의 수
    - `activation` : 적용할 activation 함수 설정
  - `[model].add(tf.keras.layers.Dense(units, activation))`
    - 모델에 Layer 추가하기
    - `units` : 레이어 안의 Node의 수
    - `activation` : 적용할 activation 함수 설정
  - `[model].compile(optimizer, loss)`
    - 모델 학습 방식을 설정하기 위한 함수
    - `optimizer` : 모델 학습 최적화 방법
    - `loss` : 손실 함수 설정
  - `[model].fit(x, y)`
    - 모델을 학습시키기 위한 함수
    - `x` : 학습 데이터
    - `y` : 학습 데이터의 label
  - `[model].evaluate(x, y)`
    - 모델을 평가하기 위한 함수
    - `x` : 테스트 데이터
    - `y` : 테스트 데이터의 label
  - `[model].predict(x)`
    - 모델로 예측을 수행하기 위한 함수
    - `x` : 예측하고자 하는 데이터

#### 2. 딥러닝 모델 구축하기

- Input Layer의 입력 형태 지정하기
  - 첫 번째 즉, Input Layer는 입력 형태에 대한 정보를 필요로 함 ([Input Layer](https://user-images.githubusercontent.com/61646760/144274003-c4b504aa-3825-44b5-aa98-caedb552681a.png))
    - `input_shape` / `input_dim` 인자 설정하기
  ```
  # model 생성 및 layer 추가
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=2, activation='sigmoid'),  # 2개의 입력 변수, 10개의 노드
    tf.keras.layers.Dense(10, activation='sigmoid'),               # 10개의 노드
    tf.keras.layers.Dense(1, activation='sigmoid'),                # 1개의 노드
  ])
  ```
- 모델에 Layer 추가하기 (add)
  ```
  # model 생성
  model = tf.keras.models.Sequential()
  
  # model에 layer 추가 (위와 동일)
  model.add(tf.keras.layers.Dense(10, input_dim=2, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  ```

#### 3. 딥러닝 모델 학습시키기
- compile, fit
  ```
  # MSE를 loss로 설정, 최적화 방식은 SGD 사용
  model.compile(loss='mean_squared_error', optimizer='SGD')
  
  # dataset에 저장된 데이터를 입력하고, epochs를 100으로 설정하고 학습
  model.fit(dataset, epochs=100)
  ```

#### 4. 평가 및 예측하기
- evaluate, predict
  ```
  # MSE를 loss로 설정, 최적화 방식은 SGD 사용
  model.compile(loss='mean_squared_error', optimizer='SGD')
  
  # dataset에 저장된 데이터를 입력하고, epochs를 100으로 설정하고 학습
  model.fit(dataset, epochs=100)
  
  # 모델 평가 및 예측하기
  model.evaluate(X_test, Y_test)
  predicted_labels_test = model.predict(X_test)
  ```

#### 텐서플로우를 활용하여 신경망 구현하기 - 모델 구현
앞에 이어서 이번 실습에서는 텐서플로우와 케라스(Keras)를 활용하여 신경망 모델을 구현해 보자.
- **케라스**는 텐서플로우 내의 신경망 모델 설계와 훈련을 위한 API이다.
  - 케라스는 연속적으로(Sequential) 레이어(Layer)들을 쌓아가며 모델을 생성하고, sklearn과 같이 한 줄의 코드로 간단하게 학습 방법 설정, 학습, 평가를 진행할 수 있다.
- 텐서플로우와 케라스를 이용해 신경망 모델을 만들기 위한 함수
  - `tf.keras.models.Sequential()`
    - 연속적으로 층을 쌓아 만드는 Sequential 모델을 위한 함수
    - `tf.keras.models` : 모델 설정
  - `tf.keras.layers.Dense(units)`
    - Dense 레이어
    - `tf.keras.layers` : 신경망 모델의 레이어를 구성하는 데 필요한 keras 함수
      - `units` : 레이어 안의 노드 수
- 예를 들어, 5개의 변수에 따른 label을 예측하는 회귀 분석 신경망을 구현하고 싶다면 아래와 같이 구현할 수 있다.
  ```
  tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,)),
    tf.keras.layers.Dense(1)
  ])
  ```
  - `input_shape` 인자에는 (입력하는 변수의 개수, )로 입력한다. 또한 회귀 분석이기에 마지막 레이어의 유닛 수는 1개로 설정한다.
  - `input_dim` 인자를 사용하면 아래와 같이 표현할 수 있다.
    ```
    tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, input_dim=5),
      tf.keras.layers.Dense(1)
    ])
    ```

```
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

X = df.drop(columns=['Sales'])
Y = df['Sales']

# 학습용 테스트용 데이터로 분리합니다.
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

# Dataset 형태로 변환합니다.
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

"""
1. tf.keras.models.Sequential()를 활용하여 신경망 모델을 생성합니다.
   자유롭게 layers를 쌓고 마지막 layers는 노드 수를 1개로 설정합니다.
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,)),  # FB, TV, Newspaper의 3개 데이터
    tf.keras.layers.Dense(10, input_shape=(3,)),  # 노드 수는 임의로
    tf.keras.layers.Dense(1)
    ])

print(model.summary())

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 10)                40        
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11        
=================================================================
Total params: 161
Trainable params: 161
Non-trainable params: 0
_________________________________________________________________
None
'''
```

#### 텐서플로우를 활용하여 신경망 구현하기 - 모델 학습
앞에 이어서 이번에는 텐서플로우와 케라스(Keras)를 활용하여 신경망 모델을 학습해 보자.
- 텐서플로우와 케라스를 이용해 신경망 모델을 학습하기 위한 함수
  - `model.compile(loss='mean_squared_error', optimizer='adam')`
    - 학습 방법 설정
    - `complie()` 메서드는 모델을 어떻게 학습할지에 대해서 설정한다.
    - loss는 회귀에서는 일반적으로 MSE인 `mean_squared_error`, 분류에서는 `sparse_categorical_crossentropy`를 주로 사용한다.
  - `model.fit(X, epochs=100, verbose=2)`
    - 학습 수행
    - `X` 데이터를 에포크를 100번으로 하여 학습한다.
    - `verbose` 인자는 학습 시, 화면에 출력되는 형태를 설정한다.
      - `0` : 표기 없음
      - `1` : 진행 바
      - `2` : 에포크당 한 줄 출력

```
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

X = df.drop(columns=['Sales'])
Y = df['Sales']

# 학습용 테스트용 데이터로 분리합니다.
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

# Dataset 형태로 변환합니다.
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)


# keras를 활용하여 신경망 모델을 생성합니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,)),
    tf.keras.layers.Dense(1)
    ])


"""
1. 학습용 데이터를 바탕으로 모델의 학습을 수행합니다.
    
step1. compile 메서드를 사용하여 최적화 모델을 설정합니다.
       loss는 mean_squared_error, optimizer는 adam으로 설정합니다.
       
step2. fit 메서드를 사용하여 Dataset으로 변환된 학습용 데이터를 학습합니다.
       epochs는 100으로 설정합니다.
"""
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_ds, epochs=100, verbose=2)

'''
Epoch 1/100
28/28 - 0s - loss: 1639.5788
Epoch 2/100
28/28 - 0s - loss: 281.0369
Epoch 3/100
28/28 - 0s - loss: 139.9947
Epoch 4/100
28/28 - 0s - loss: 98.7728
Epoch 5/100
28/28 - 0s - loss: 66.9869
Epoch 6/100
28/28 - 0s - loss: 46.4901
Epoch 7/100
28/28 - 0s - loss: 33.6584
Epoch 8/100
28/28 - 0s - loss: 24.5128
Epoch 9/100
28/28 - 0s - loss: 17.3948
Epoch 10/100
28/28 - 0s - loss: 15.0934
Epoch 11/100
28/28 - 0s - loss: 13.0036
Epoch 12/100
28/28 - 0s - loss: 10.7213
Epoch 13/100
28/28 - 0s - loss: 9.2237
Epoch 14/100
28/28 - 0s - loss: 8.2051
Epoch 15/100
28/28 - 0s - loss: 7.4424
Epoch 16/100
28/28 - 0s - loss: 6.2867
Epoch 17/100
28/28 - 0s - loss: 6.0592
Epoch 18/100
28/28 - 0s - loss: 5.6608
Epoch 19/100
28/28 - 0s - loss: 4.5156
Epoch 20/100
28/28 - 0s - loss: 5.1645
Epoch 21/100
28/28 - 0s - loss: 4.7275
Epoch 22/100
28/28 - 0s - loss: 4.7179
Epoch 23/100
28/28 - 0s - loss: 4.5683
Epoch 24/100
28/28 - 0s - loss: 4.5556
Epoch 25/100
28/28 - 0s - loss: 4.4095
Epoch 26/100
28/28 - 0s - loss: 4.2198
Epoch 27/100
28/28 - 0s - loss: 4.4354
Epoch 28/100
28/28 - 0s - loss: 4.4298
Epoch 29/100
28/28 - 0s - loss: 4.3224
Epoch 30/100
28/28 - 0s - loss: 4.2932
Epoch 31/100
28/28 - 0s - loss: 4.2893
Epoch 32/100
28/28 - 0s - loss: 4.2396
Epoch 33/100
28/28 - 0s - loss: 4.2788
Epoch 34/100
28/28 - 0s - loss: 4.1441
Epoch 35/100
28/28 - 0s - loss: 4.0707
Epoch 36/100
28/28 - 0s - loss: 4.0699
Epoch 37/100
28/28 - 0s - loss: 4.2229
Epoch 38/100
28/28 - 0s - loss: 4.3146
Epoch 39/100
28/28 - 0s - loss: 4.2231
Epoch 40/100
28/28 - 0s - loss: 4.2952
Epoch 41/100
28/28 - 0s - loss: 4.1489
Epoch 42/100
28/28 - 0s - loss: 4.2076
Epoch 43/100
28/28 - 0s - loss: 4.0038
Epoch 44/100
28/28 - 0s - loss: 4.1169
Epoch 45/100
28/28 - 0s - loss: 4.3562
Epoch 46/100
28/28 - 0s - loss: 4.1704
Epoch 47/100
28/28 - 0s - loss: 4.2968
Epoch 48/100
28/28 - 0s - loss: 4.1475
Epoch 49/100
28/28 - 0s - loss: 4.0728
Epoch 50/100
28/28 - 0s - loss: 4.1516
Epoch 51/100
28/28 - 0s - loss: 3.9765
Epoch 52/100
28/28 - 0s - loss: 4.2054
Epoch 53/100
28/28 - 0s - loss: 4.0494
Epoch 54/100
28/28 - 0s - loss: 3.9231
Epoch 55/100
28/28 - 0s - loss: 4.1474
Epoch 56/100
28/28 - 0s - loss: 4.1947
Epoch 57/100
28/28 - 0s - loss: 3.7425
Epoch 58/100
28/28 - 0s - loss: 3.6219
Epoch 59/100
28/28 - 0s - loss: 4.1503
Epoch 60/100
28/28 - 0s - loss: 4.1945
Epoch 61/100
28/28 - 0s - loss: 3.5830
Epoch 62/100
28/28 - 0s - loss: 4.0271
Epoch 63/100
28/28 - 0s - loss: 4.0436
Epoch 64/100
28/28 - 0s - loss: 4.0728
Epoch 65/100
28/28 - 0s - loss: 4.0300
Epoch 66/100
28/28 - 0s - loss: 3.9253
Epoch 67/100
28/28 - 0s - loss: 3.9704
Epoch 68/100
28/28 - 0s - loss: 3.9507
Epoch 69/100
28/28 - 0s - loss: 4.1810
Epoch 70/100
28/28 - 0s - loss: 4.0647
Epoch 71/100
28/28 - 0s - loss: 4.1985
Epoch 72/100
28/28 - 0s - loss: 4.1481
Epoch 73/100
28/28 - 0s - loss: 4.0116
Epoch 74/100
28/28 - 0s - loss: 3.9494
Epoch 75/100
28/28 - 0s - loss: 3.9130
Epoch 76/100
28/28 - 0s - loss: 3.9064
Epoch 77/100
28/28 - 0s - loss: 3.8694
Epoch 78/100
28/28 - 0s - loss: 3.8523
Epoch 79/100
28/28 - 0s - loss: 3.9759
Epoch 80/100
28/28 - 0s - loss: 4.0342
Epoch 81/100
28/28 - 0s - loss: 3.9083
Epoch 82/100
28/28 - 0s - loss: 4.1890
Epoch 83/100
28/28 - 0s - loss: 3.9713
Epoch 84/100
28/28 - 0s - loss: 4.1247
Epoch 85/100
28/28 - 0s - loss: 4.4938
Epoch 86/100
28/28 - 0s - loss: 4.0068
Epoch 87/100
28/28 - 0s - loss: 3.8684
Epoch 88/100
28/28 - 0s - loss: 3.7362
Epoch 89/100
28/28 - 0s - loss: 3.8032
Epoch 90/100
28/28 - 0s - loss: 3.8723
Epoch 91/100
28/28 - 0s - loss: 3.8965
Epoch 92/100
28/28 - 0s - loss: 3.7493
Epoch 93/100
28/28 - 0s - loss: 3.6952
Epoch 94/100
28/28 - 0s - loss: 3.8224
Epoch 95/100
28/28 - 0s - loss: 3.8564
Epoch 96/100
28/28 - 0s - loss: 3.6585
Epoch 97/100
28/28 - 0s - loss: 3.5836
Epoch 98/100
28/28 - 0s - loss: 3.6144
Epoch 99/100
28/28 - 0s - loss: 3.7437
Epoch 100/100
28/28 - 0s - loss: 3.9819
'''
```

#### 텐서플로우를 활용하여 신경망 구현하기 - 모델 평가 및 예측
앞에 이어서 이번에는 학습된 신경망을 모델을 평가하고 예측해 보자.
- 텐서플로우를 이용해 신경망 모델을 평가 및 예측을 위한 함수
  - `model.evaluate(X, Y)`
    - 평가 방법
    - `evaluate()` 메서드는 학습된 모델을 바탕으로 입력한 feature 데이터 `X`와 label `Y`의 loss 값과 metrics 값을 출력한다.
    - 아래에서는 metrics를 compile에서 설정하지 않았지만, 분류에서는 일반적으로 `accuracy`를 사용하여 evaluate 사용 시, 2개의 아웃풋을 리턴한다.
  - `model.predict(X)`
    - 예측 방법
    - `X` 데이터의 예측 label 값을 출력한다.

```
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

X = df.drop(columns=['Sales'])
Y = df['Sales']

# 학습용 테스트용 데이터로 분리합니다.
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

# Dataset 형태로 변환합니다.
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

# keras를 활용하여 신경망 모델을 생성합니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,)),
    tf.keras.layers.Dense(1)
    ])

# 학습용 데이터를 바탕으로 모델의 학습을 수행합니다.
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_ds, epochs=100, verbose=2)

'''
Epoch 1/100
28/28 - 0s - loss: 1639.5788
Epoch 2/100
28/28 - 0s - loss: 281.0369
Epoch 3/100
28/28 - 0s - loss: 139.9947
Epoch 4/100
28/28 - 0s - loss: 98.7728
Epoch 5/100
28/28 - 0s - loss: 66.9869
Epoch 6/100
28/28 - 0s - loss: 46.4901
Epoch 7/100
28/28 - 0s - loss: 33.6584
Epoch 8/100
28/28 - 0s - loss: 24.5128
Epoch 9/100
28/28 - 0s - loss: 17.3948
Epoch 10/100
28/28 - 0s - loss: 15.0934
Epoch 11/100
28/28 - 0s - loss: 13.0036
Epoch 12/100
28/28 - 0s - loss: 10.7213
Epoch 13/100
28/28 - 0s - loss: 9.2237
Epoch 14/100
28/28 - 0s - loss: 8.2051
Epoch 15/100
28/28 - 0s - loss: 7.4424
Epoch 16/100
28/28 - 0s - loss: 6.2867
Epoch 17/100
28/28 - 0s - loss: 6.0592
Epoch 18/100
28/28 - 0s - loss: 5.6608
Epoch 19/100
28/28 - 0s - loss: 4.5156
Epoch 20/100
28/28 - 0s - loss: 5.1645
Epoch 21/100
28/28 - 0s - loss: 4.7275
Epoch 22/100
28/28 - 0s - loss: 4.7179
Epoch 23/100
28/28 - 0s - loss: 4.5683
Epoch 24/100
28/28 - 0s - loss: 4.5556
Epoch 25/100
28/28 - 0s - loss: 4.4095
Epoch 26/100
28/28 - 0s - loss: 4.2198
Epoch 27/100
28/28 - 0s - loss: 4.4354
Epoch 28/100
28/28 - 0s - loss: 4.4298
Epoch 29/100
28/28 - 0s - loss: 4.3224
Epoch 30/100
28/28 - 0s - loss: 4.2932
Epoch 31/100
28/28 - 0s - loss: 4.2893
Epoch 32/100
28/28 - 0s - loss: 4.2396
Epoch 33/100
28/28 - 0s - loss: 4.2788
Epoch 34/100
28/28 - 0s - loss: 4.1441
Epoch 35/100
28/28 - 0s - loss: 4.0707
Epoch 36/100
28/28 - 0s - loss: 4.0699
Epoch 37/100
28/28 - 0s - loss: 4.2229
Epoch 38/100
28/28 - 0s - loss: 4.3146
Epoch 39/100
28/28 - 0s - loss: 4.2231
Epoch 40/100
28/28 - 0s - loss: 4.2952
Epoch 41/100
28/28 - 0s - loss: 4.1489
Epoch 42/100
28/28 - 0s - loss: 4.2076
Epoch 43/100
28/28 - 0s - loss: 4.0038
Epoch 44/100
28/28 - 0s - loss: 4.1169
Epoch 45/100
28/28 - 0s - loss: 4.3562
Epoch 46/100
28/28 - 0s - loss: 4.1704
Epoch 47/100
28/28 - 0s - loss: 4.2968
Epoch 48/100
28/28 - 0s - loss: 4.1475
Epoch 49/100
28/28 - 0s - loss: 4.0728
Epoch 50/100
28/28 - 0s - loss: 4.1516
Epoch 51/100
28/28 - 0s - loss: 3.9765
Epoch 52/100
28/28 - 0s - loss: 4.2054
Epoch 53/100
28/28 - 0s - loss: 4.0494
Epoch 54/100
28/28 - 0s - loss: 3.9231
Epoch 55/100
28/28 - 0s - loss: 4.1474
Epoch 56/100
28/28 - 0s - loss: 4.1947
Epoch 57/100
28/28 - 0s - loss: 3.7425
Epoch 58/100
28/28 - 0s - loss: 3.6219
Epoch 59/100
28/28 - 0s - loss: 4.1503
Epoch 60/100
28/28 - 0s - loss: 4.1945
Epoch 61/100
28/28 - 0s - loss: 3.5830
Epoch 62/100
28/28 - 0s - loss: 4.0271
Epoch 63/100
28/28 - 0s - loss: 4.0436
Epoch 64/100
28/28 - 0s - loss: 4.0728
Epoch 65/100
28/28 - 0s - loss: 4.0300
Epoch 66/100
28/28 - 0s - loss: 3.9253
Epoch 67/100
28/28 - 0s - loss: 3.9704
Epoch 68/100
28/28 - 0s - loss: 3.9507
Epoch 69/100
28/28 - 0s - loss: 4.1810
Epoch 70/100
28/28 - 0s - loss: 4.0647
Epoch 71/100
28/28 - 0s - loss: 4.1985
Epoch 72/100
28/28 - 0s - loss: 4.1481
Epoch 73/100
28/28 - 0s - loss: 4.0116
Epoch 74/100
28/28 - 0s - loss: 3.9494
Epoch 75/100
28/28 - 0s - loss: 3.9130
Epoch 76/100
28/28 - 0s - loss: 3.9064
Epoch 77/100
28/28 - 0s - loss: 3.8694
Epoch 78/100
28/28 - 0s - loss: 3.8523
Epoch 79/100
28/28 - 0s - loss: 3.9759
Epoch 80/100
28/28 - 0s - loss: 4.0342
Epoch 81/100
28/28 - 0s - loss: 3.9083
Epoch 82/100
28/28 - 0s - loss: 4.1890
Epoch 83/100
28/28 - 0s - loss: 3.9713
Epoch 84/100
28/28 - 0s - loss: 4.1247
Epoch 85/100
28/28 - 0s - loss: 4.4938
Epoch 86/100
28/28 - 0s - loss: 4.0068
Epoch 87/100
28/28 - 0s - loss: 3.8684
Epoch 88/100
28/28 - 0s - loss: 3.7362
Epoch 89/100
28/28 - 0s - loss: 3.8032
Epoch 90/100
28/28 - 0s - loss: 3.8723
Epoch 91/100
28/28 - 0s - loss: 3.8965
Epoch 92/100
28/28 - 0s - loss: 3.7493
Epoch 93/100
28/28 - 0s - loss: 3.6952
Epoch 94/100
28/28 - 0s - loss: 3.8224
Epoch 95/100
28/28 - 0s - loss: 3.8564
Epoch 96/100
28/28 - 0s - loss: 3.6585
Epoch 97/100
28/28 - 0s - loss: 3.5836
Epoch 98/100
28/28 - 0s - loss: 3.6144
Epoch 99/100
28/28 - 0s - loss: 3.7437
Epoch 100/100
28/28 - 0s - loss: 3.9819
'''

"""
1. evaluate 메서드를 사용하여 테스트용 데이터의 loss 값을 계산합니다.
"""
loss = model.evaluate(test_X, test_Y, verbose=0)

"""
2. predict 메서드를 사용하여 테스트용 데이터의 예측값을 계산합니다.
"""
predictions = model.predict(test_X)

# 결과를 출력합니다.
print("테스트 데이터의 Loss 값:", loss)
for i in range(5):
    print("%d 번째 테스트 데이터의 실제값: %f" % (i, test_Y.iloc[i]))
    print("%d 번째 테스트 데이터의 예측값: %f" % (i, predictions[i][0]))

'''
테스트 데이터의 Loss 값: 3.6781873226165773
0 번째 테스트 데이터의 실제값: 6.600000
0 번째 테스트 데이터의 예측값: 10.392957
1 번째 테스트 데이터의 실제값: 20.700000
1 번째 테스트 데이터의 예측값: 19.001699
2 번째 테스트 데이터의 실제값: 17.200000
2 번째 테스트 데이터의 예측값: 16.554482
3 번째 테스트 데이터의 실제값: 19.400000
3 번째 테스트 데이터의 예측값: 18.733145
4 번째 테스트 데이터의 실제값: 21.800000
4 번째 테스트 데이터의 예측값: 20.184616
'''
```

- `evaluate` 메서드를 사용하여 테스트용 데이터의 loss 값을 계산하고 `loss`에 저장한다.
- `predict` 메서드를 사용하여 테스트용 데이터의 예측값을 계산하고 `predictions`에 저장한다.

#### 신경망 모델로 분류하기
이번에는 Iris 데이터가 주어졌을 때 붓꽃의 종류를 분류하는 신경망 모델을 구현해 보자.
- Iris 데이터는 아래와 같이 꽃받침 길이, 꽃받침 넓이, 꽃잎 길이, 꽃잎 넓이 네 가지 변수와 세 종류의 붓꽃 클래스로 구성되어 있다.
  ![image](https://user-images.githubusercontent.com/61646760/144714349-2f2b26c7-f3c0-44e8-926c-dfa527cdb310.png)
- 분류를 위한 텐서플로우 신경망 모델 함수
  - 모델 구현 (5개의 범주를 갖는 label 예시)
    ```
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, input_dim=4),
      tf.keras.layers.Dense(5, activation='softmax')
      ])
    ```
    - 분류 모델에서는 마지막 레이어에 분류 데이터의 label 범주의 개수만큼 노드를 설정한다.
    - 추가로 `activation` 인자로 'softmax'를 설정한다.
  - 학습 방법
    - `model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])`
      - 분류에서는 일반적으로 `loss`를 'sparse_categorical_crossentropy'로 사용한다.
      - `metrics` 인자는 에포크마다 계산되는 평가 지표를 의미한다. 정확도를 의미하는 'accuracy'를 입력하면 에포크마다 accuracy를 계산하여 출력한다.

```
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(100)
tf.random.set_seed(100)

# sklearn에 저장된 데이터를 불러옵니다.
X, Y = load_iris(return_X_y = True)

# DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']

# 학습용 평가용 데이터로 분리합니다.
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

# Dataset 형태로 변환합니다.
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

"""
1. keras를 활용하여 신경망 모델을 생성합니다.
   3가지 범주를 갖는 label 데이터를 분류하기 위해서 마지막 레이어 노드를 아래와 같이 설정합니다.
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=4),
    tf.keras.layers.Dense(3, activation='softmax')  # 3개 범주의 label
    ])

# 학습용 데이터를 바탕으로 모델의 학습을 수행합니다.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_ds, epochs=100, verbose=2)

'''
Epoch 1/100
24/24 - 0s - loss: 1.7421 - accuracy: 0.3750
Epoch 2/100
24/24 - 0s - loss: 1.3935 - accuracy: 0.5500
Epoch 3/100
24/24 - 0s - loss: 1.2528 - accuracy: 0.4833
Epoch 4/100
24/24 - 0s - loss: 1.1385 - accuracy: 0.5000
Epoch 5/100
24/24 - 0s - loss: 1.0317 - accuracy: 0.5000
Epoch 6/100
24/24 - 0s - loss: 0.9339 - accuracy: 0.5083
Epoch 7/100
24/24 - 0s - loss: 0.8466 - accuracy: 0.5083
Epoch 8/100
24/24 - 0s - loss: 0.7704 - accuracy: 0.5333
Epoch 9/100
24/24 - 0s - loss: 0.7053 - accuracy: 0.8167
Epoch 10/100
24/24 - 0s - loss: 0.6502 - accuracy: 0.8667
Epoch 11/100
24/24 - 0s - loss: 0.6042 - accuracy: 0.8833
Epoch 12/100
24/24 - 0s - loss: 0.5657 - accuracy: 0.8917
Epoch 13/100
24/24 - 0s - loss: 0.5335 - accuracy: 0.9167
Epoch 14/100
24/24 - 0s - loss: 0.5063 - accuracy: 0.9167
Epoch 15/100
24/24 - 0s - loss: 0.4831 - accuracy: 0.9167
Epoch 16/100
24/24 - 0s - loss: 0.4632 - accuracy: 0.9250
Epoch 17/100
24/24 - 0s - loss: 0.4458 - accuracy: 0.9333
Epoch 18/100
24/24 - 0s - loss: 0.4304 - accuracy: 0.9333
Epoch 19/100
24/24 - 0s - loss: 0.4167 - accuracy: 0.9333
Epoch 20/100
24/24 - 0s - loss: 0.4043 - accuracy: 0.9417
Epoch 21/100
24/24 - 0s - loss: 0.3929 - accuracy: 0.9417
Epoch 22/100
24/24 - 0s - loss: 0.3825 - accuracy: 0.9417
Epoch 23/100
24/24 - 0s - loss: 0.3728 - accuracy: 0.9417
Epoch 24/100
24/24 - 0s - loss: 0.3637 - accuracy: 0.9417
Epoch 25/100
24/24 - 0s - loss: 0.3552 - accuracy: 0.9500
Epoch 26/100
24/24 - 0s - loss: 0.3471 - accuracy: 0.9667
Epoch 27/100
24/24 - 0s - loss: 0.3393 - accuracy: 0.9667
Epoch 28/100
24/24 - 0s - loss: 0.3319 - accuracy: 0.9667
Epoch 29/100
24/24 - 0s - loss: 0.3248 - accuracy: 0.9667
Epoch 30/100
24/24 - 0s - loss: 0.3179 - accuracy: 0.9667
Epoch 31/100
24/24 - 0s - loss: 0.3113 - accuracy: 0.9667
Epoch 32/100
24/24 - 0s - loss: 0.3049 - accuracy: 0.9667
Epoch 33/100
24/24 - 0s - loss: 0.2986 - accuracy: 0.9583
Epoch 34/100
24/24 - 0s - loss: 0.2926 - accuracy: 0.9583
Epoch 35/100
24/24 - 0s - loss: 0.2867 - accuracy: 0.9583
Epoch 36/100
24/24 - 0s - loss: 0.2810 - accuracy: 0.9583
Epoch 37/100
24/24 - 0s - loss: 0.2754 - accuracy: 0.9583
Epoch 38/100
24/24 - 0s - loss: 0.2700 - accuracy: 0.9583
Epoch 39/100
24/24 - 0s - loss: 0.2647 - accuracy: 0.9583
Epoch 40/100
24/24 - 0s - loss: 0.2595 - accuracy: 0.9583
Epoch 41/100
24/24 - 0s - loss: 0.2545 - accuracy: 0.9583
Epoch 42/100
24/24 - 0s - loss: 0.2496 - accuracy: 0.9583
Epoch 43/100
24/24 - 0s - loss: 0.2448 - accuracy: 0.9583
Epoch 44/100
24/24 - 0s - loss: 0.2402 - accuracy: 0.9583
Epoch 45/100
24/24 - 0s - loss: 0.2356 - accuracy: 0.9667
Epoch 46/100
24/24 - 0s - loss: 0.2312 - accuracy: 0.9667
Epoch 47/100
24/24 - 0s - loss: 0.2269 - accuracy: 0.9667
Epoch 48/100
24/24 - 0s - loss: 0.2227 - accuracy: 0.9667
Epoch 49/100
24/24 - 0s - loss: 0.2186 - accuracy: 0.9667
Epoch 50/100
24/24 - 0s - loss: 0.2147 - accuracy: 0.9667
Epoch 51/100
24/24 - 0s - loss: 0.2108 - accuracy: 0.9667
Epoch 52/100
24/24 - 0s - loss: 0.2070 - accuracy: 0.9667
Epoch 53/100
24/24 - 0s - loss: 0.2034 - accuracy: 0.9667
Epoch 54/100
24/24 - 0s - loss: 0.1998 - accuracy: 0.9667
Epoch 55/100
24/24 - 0s - loss: 0.1963 - accuracy: 0.9667
Epoch 56/100
24/24 - 0s - loss: 0.1930 - accuracy: 0.9667
Epoch 57/100
24/24 - 0s - loss: 0.1897 - accuracy: 0.9667
Epoch 58/100
24/24 - 0s - loss: 0.1865 - accuracy: 0.9667
Epoch 59/100
24/24 - 0s - loss: 0.1834 - accuracy: 0.9750
Epoch 60/100
24/24 - 0s - loss: 0.1804 - accuracy: 0.9750
Epoch 61/100
24/24 - 0s - loss: 0.1775 - accuracy: 0.9750
Epoch 62/100
24/24 - 0s - loss: 0.1746 - accuracy: 0.9750
Epoch 63/100
24/24 - 0s - loss: 0.1718 - accuracy: 0.9750
Epoch 64/100
24/24 - 0s - loss: 0.1692 - accuracy: 0.9750
Epoch 65/100
24/24 - 0s - loss: 0.1666 - accuracy: 0.9750
Epoch 66/100
24/24 - 0s - loss: 0.1640 - accuracy: 0.9750
Epoch 67/100
24/24 - 0s - loss: 0.1615 - accuracy: 0.9750
Epoch 68/100
24/24 - 0s - loss: 0.1592 - accuracy: 0.9750
Epoch 69/100
24/24 - 0s - loss: 0.1568 - accuracy: 0.9750
Epoch 70/100
24/24 - 0s - loss: 0.1546 - accuracy: 0.9750
Epoch 71/100
24/24 - 0s - loss: 0.1524 - accuracy: 0.9750
Epoch 72/100
24/24 - 0s - loss: 0.1502 - accuracy: 0.9750
Epoch 73/100
24/24 - 0s - loss: 0.1482 - accuracy: 0.9750
Epoch 74/100
24/24 - 0s - loss: 0.1461 - accuracy: 0.9750
Epoch 75/100
24/24 - 0s - loss: 0.1442 - accuracy: 0.9750
Epoch 76/100
24/24 - 0s - loss: 0.1423 - accuracy: 0.9750
Epoch 77/100
24/24 - 0s - loss: 0.1404 - accuracy: 0.9750
Epoch 78/100
24/24 - 0s - loss: 0.1386 - accuracy: 0.9750
Epoch 79/100
24/24 - 0s - loss: 0.1369 - accuracy: 0.9750
Epoch 80/100
24/24 - 0s - loss: 0.1351 - accuracy: 0.9750
Epoch 81/100
24/24 - 0s - loss: 0.1335 - accuracy: 0.9750
Epoch 82/100
24/24 - 0s - loss: 0.1319 - accuracy: 0.9750
Epoch 83/100
24/24 - 0s - loss: 0.1303 - accuracy: 0.9750
Epoch 84/100
24/24 - 0s - loss: 0.1288 - accuracy: 0.9750
Epoch 85/100
24/24 - 0s - loss: 0.1273 - accuracy: 0.9750
Epoch 86/100
24/24 - 0s - loss: 0.1258 - accuracy: 0.9750
Epoch 87/100
24/24 - 0s - loss: 0.1244 - accuracy: 0.9750
Epoch 88/100
24/24 - 0s - loss: 0.1231 - accuracy: 0.9750
Epoch 89/100
24/24 - 0s - loss: 0.1217 - accuracy: 0.9750
Epoch 90/100
24/24 - 0s - loss: 0.1204 - accuracy: 0.9750
Epoch 91/100
24/24 - 0s - loss: 0.1192 - accuracy: 0.9750
Epoch 92/100
24/24 - 0s - loss: 0.1179 - accuracy: 0.9750
Epoch 93/100
24/24 - 0s - loss: 0.1167 - accuracy: 0.9750
Epoch 94/100
24/24 - 0s - loss: 0.1156 - accuracy: 0.9750
Epoch 95/100
24/24 - 0s - loss: 0.1144 - accuracy: 0.9750
Epoch 96/100
24/24 - 0s - loss: 0.1133 - accuracy: 0.9750
Epoch 97/100
24/24 - 0s - loss: 0.1122 - accuracy: 0.9750
Epoch 98/100
24/24 - 0s - loss: 0.1112 - accuracy: 0.9750
Epoch 99/100
24/24 - 0s - loss: 0.1102 - accuracy: 0.9750
Epoch 100/100
24/24 - 0s - loss: 0.1091 - accuracy: 0.9750

30/30 [==============================] - 0s 480us/sample - loss: 0.1161 - accuracy: 1.0000
'''

# 테스트용 데이터를 바탕으로 학습된 모델을 평가합니다.
loss, acc = model.evaluate(test_X, test_Y)

# 테스트용 데이터의 예측값을 구합니다.
predictions = model.predict(test_X)

# 결과를 출력합니다.
print("테스트 데이터의 Accuracy 값:", acc)
for i in range(5):
    print("%d 번째 테스트 데이터의 실제값: %d" % (i, test_Y.iloc[i]))
    print("%d 번째 테스트 데이터의 예측값: %d" % (i, np.argmax(predictions[i])))

'''
테스트 데이터의 Accuracy 값: 1.0
0 번째 테스트 데이터의 실제값: 1
0 번째 테스트 데이터의 예측값: 1
1 번째 테스트 데이터의 실제값: 0
1 번째 테스트 데이터의 예측값: 0
2 번째 테스트 데이터의 실제값: 2
2 번째 테스트 데이터의 예측값: 2
3 번째 테스트 데이터의 실제값: 1
3 번째 테스트 데이터의 예측값: 1
4 번째 테스트 데이터의 실제값: 1
4 번째 테스트 데이터의 예측값: 1
'''
```

## 03. 다양한 신경망
### 이미지 처리를 위한 데이터 전 처리
- 이미지 처리 기술 예시
  - 얼굴 인식 카메라
  - 화질 개선(Super Resolution)
  - 이미지 자동 태깅
- 컴퓨터는 각각의 픽셀 값을 가진 숫자 배열로 이미지를 인식  

#### 이미지 전 처리하기
![image](https://user-images.githubusercontent.com/61646760/144873366-a60be7da-2d67-4607-9103-1c2aa58721b9.png)
- 모두 같은 크기를 갖는 이미지로 통일
  1. 가로, 세로 픽셀 사이즈를 표현하는 **해상도 통일**
  2. 색을 표현하는 방식 통일 `예) RGB, HSV, Gray-scale, Binary, ...`

### MNIST 분류 CNN 모델 - 데이터 전 처리
신경망을 이용한 학습을 시작할 때 대부분 MNIST를 접하게 된다.
- MNIST는 손글씨로 된 사진을 모아 둔 데이터이다.  
  ![image](https://user-images.githubusercontent.com/61646760/145053770-da9fe4e8-e3cd-41d2-b0de-f13e21c5b82e.png)
- 손으로 쓴 0부터 9까지의 글자들이 있고, 이 데이터를 사용해서 신경망을 학습시키고, 학습 결과가 손글씨를 인식할 수 있는지 검증한다.
- 이번 실습에서는 우선 이미지 데이터를 출력하고 그 형태를 확인하여 CNN 모델에 적용할 수 있도록 데이터 전 처리를 수행한다.

#### CNN을 위한 데이터 전 처리
- MNIST 데이터는 이미지 데이터이지만 가로 길이와 세로 길이만 존재하는 2차원 데이터이다.
- CNN 모델은 채널(RGB 혹은 흑백)까지 고려한 3차원 데이터를 입력으로 받으므로, 채널 차원을 추가해 아래와 같이 데이터의 모양(shape)을 바꿔 준다.
  - `[데이터 수, 가로 길이, 세로 길이] → [데이터 수, 가로 길이, 세로 길이, 채널 수]`
- `tf.expand_dims(data, axis)`
  - 차원 추가 함수
  - Tensor 배열 데이터에서 마지막 축(`axis`)에 해당하는 곳에 차원 하나를 추가할 수 있는 함수
    - `axis`에 -1을 넣으면 어떤 `data`가 들어오든 마지막 축의 index를 의미한다.

```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)


# MNIST 데이터 세트를 불러옵니다.
mnist = tf.keras.datasets.mnist

# MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    

# Train 데이터 5000개와 Test 데이터 1000개를 사용합니다.
train_images, train_labels = train_images[:5000], train_labels[:5000]
test_images, test_labels = test_images[:1000], test_labels[:1000]


print("원본 학습용 이미지 데이터 형태:", train_images.shape)
print("원본 평가용 이미지 데이터 형태:", test_images.shape)
print("원본 학습용 label 데이터:", train_labels)

'''
원본 학습용 이미지 데이터 형태: (5000, 28, 28)
원본 평가용 이미지 데이터 형태: (1000, 28, 28)
원본 학습용 label 데이터: [5 0 4 ... 2 1 2]
'''

# 첫 번째 샘플 데이터를 출력합니다.
plt.figure(figsize=(10, 10))
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.title("Training Data Sample")
plt.savefig("sample1.png")
elice_utils.send_image("sample1.png")

# 9개의 학습용 샘플 데이터를 출력합니다.
class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig("sample2.png")
elice_utils.send_image("sample2.png")

"""
1. CNN 모델의 입력으로 사용할 수 있도록 (샘플개수, 가로픽셀, 세로픽셀, 1) 형태로 변환합니다.
"""
train_images = tf.expand_dims(train_images, -1)  # axis = -1
test_images = tf.expand_dims(test_images, -1)

print("변환한 학습용 이미지 데이터 형태:", train_images.shape)
print("변환한 평가용 이미지 데이터 형태:", test_images.shape)

'''
변환한 학습용 이미지 데이터 형태: (5000, 28, 28, 1)
변환한 평가용 이미지 데이터 형태: (1000, 28, 28, 1)
'''
```
- [sample1.png](https://user-images.githubusercontent.com/61646760/145058333-375f7774-fb91-4e4e-8f8b-b86d0ec63eae.png)
- [sample2.png](https://user-images.githubusercontent.com/61646760/145058460-278391d4-daee-4e7a-8739-5c01d7b3c27e.png)
- 학습용 및 평가용 데이터를 CNN 모델의 입력으로 사용할 수 있도록 (샘플 개수, 가로 픽셀, 세로 픽셀, 1) 형태로 변환한다.
  - `tf.expand_dims` 함수를 활용하여 `train_images`, `test_images` 데이터의 형태를 변환하고 각각 `train_images`, `test_images`에 저장한다.

### 이미지 처리를 위한 딥러닝 모델
- 기존 다층 퍼셉트론 기반 신경망의 이미지 처리는 극도로 많은 수의 패러미터가 필요하다.
  - 이미지에 변화가 있다면?
- **합성곱 신경망(Convolution Neural Network: CNN)**
  - 합성곱 신경망은 크게 합성곱 층과(Convolution layer)와 풀링 층(Pooling layer)으로 구성되어, 합성곱 연산과 풀링 연산을 통해 이미지 처리에 탁월한 성능을 보이는 신경망  
  ![image](https://user-images.githubusercontent.com/61646760/145232338-a9e2f76f-211c-4861-9b81-07ed48b0e060.png)
  - 작은 필터를 순환시키는 방식
  - 이미지의 패턴이 아닌 특징을 중점으로 인식
    - [`예) 고양이`](https://user-images.githubusercontent.com/61646760/145231737-2afa6735-a8a8-4bff-a5d0-04085dac521b.png)
    - [자세히 읽기 :  딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/64066)  
  - CNN 과정
    1. 이미지에 어떠한 특징이 있는지를 구하는 과정
        - 필터가 이미지를 이동하며 새로운 이미지(피쳐맵)를 생성
          - [이미지 보기](https://user-images.githubusercontent.com/61646760/145232511-c9846f67-468a-402a-bbf6-9a5e8835e9a4.png)
    2. 피쳐맵의 크기 변형: Padding, Striding
        - **Padding**: 원본 이미지의 상하좌우에 한 줄씩 추가
        - **Striding**: 필터를 이동시키는 거리(Stride) 설정
          - [이미지 보기](https://user-images.githubusercontent.com/61646760/145233035-d806c44f-bd54-4dac-9381-e8c7f4bb2f75.png)
    3. Pooling Layer
        - 이미지 왜곡의 영향(노이즈)을 축소하는 과정
          - [이미지 보기](https://user-images.githubusercontent.com/61646760/145233353-7a3a132a-31a8-428f-ac9a-9f29d83751d8.png)
    4. Fully Connected Layer
        - 추출된 특징을 사용하여 이미지를 분류
          - 분류를 위한 Softmax 활성화 함수
            - [이미지 보기](https://user-images.githubusercontent.com/61646760/145233672-69fa3608-ed97-4c50-ac26-35f4db552879.png)
  - 정리  
    ![image](https://user-images.githubusercontent.com/61646760/145233960-2ba6f4d2-f7ea-433b-bf6c-ee9e9c10de54.png)
    1. Convolution Layer는 특징을 찾아내고, Pooling Layer는 처리할 맵(이미지) 크기를 줄여준다.
    2. 이를 N번 반복한다.
    3. 반복할 때마다 줄어든 영역에서의 특징을 찾게 되고, 영역의 크기는 작아졌기 때문에 빠른 학습이 가능해진다.
  - 합성곱 신경망 기반 다양한 이미지 처리 기술
    - Object detection & segmentation
    - Super resolution (SR)

### MNIST 분류 CNN 모델 - 모델 구현
앞에 이어서 이번에는 CNN 모델을 구현하고 학습해 보자.
- Keras에서 CNN 모델을 만들기 위해 필요한 함수
  1. **CNN 레이어**
      - 입력 이미지의 특징, 즉 처리할 특징 맵(map)을 추출하는 레이어
      - `tf.keras.layers.Conv2D(filters, kernel_size, activation, padding)`
        - `filters` : 필터(커널) 개수
        - `kernel_size` : 필터(커널)의 크기
        - `activation` : 활성화 함수
        - `padding` : 이미지가 필터를 거칠 때 그 크기가 줄어드는 것을 방지하기 위해 가장자리에 0의 값을 가지는 픽셀을 넣을지 말지를 결정하는 변수.
          - `SAME` 또는 `VALID`
  2. **Maxpool 레이어**
      - 처리할 특징 맵(map)의 크기를 줄여주는 레이어
      - `tf.keras.layers.MaxPool2D(padding)`
        - `padding` : `SAME` 또는 `VALID` 
  3. **Flatten 레이어**
      - Convolution layer 또는 MaxPooling layer의 결과는 N차원의 텐서 형태인데, 이를 1차원으로 평평하게 만들어 주는 함수
      - `tf.keras.layers.Flatten()`
  4. **Dense 레이어**
      - `tf.keras.layers.Dense(node, activation)`
        - `node` : 노드(뉴런) 개수
        - `activation` : 활성화 함수

```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from visual import *
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)


# MNIST 데이터 세트를 불러옵니다.
mnist = tf.keras.datasets.mnist

# MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    

# Train 데이터 5000개와 Test 데이터 1000개를 사용합니다.
train_images, train_labels = train_images[:5000], train_labels[:5000]
test_images, test_labels = test_images[:1000], test_labels[:1000]

# CNN 모델의 입력으로 사용할 수 있도록 (샘플개수, 가로픽셀, 세로픽셀, 1) 형태로 변환합니다.
train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)


"""
1. CNN 모델을 설정합니다.
   분류 모델에 맞게 마지막 레이어의 노드 수는 10개, activation 함수는 'softmax'로 설정합니다.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME', input_shape = (28,28,1)),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')  # 노드 수 10개
])

# CNN 모델 구조를 출력합니다.
print(model.summary())

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 32)          9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                32832     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 52,298
Trainable params: 52,298
Non-trainable params: 0
_________________________________________________________________
None
'''

# CNN 모델의 학습 방법을 설정합니다.
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
              
# 학습을 수행합니다. 
history = model.fit(train_images, train_labels, epochs = 20, batch_size = 512)

'''
Epoch 20/20

 512/5000 [==>...........................] - ETA: 1s - loss: 0.1084 - accuracy: 0.9668
1024/5000 [=====>........................] - ETA: 1s - loss: 0.1283 - accuracy: 0.9580
1536/5000 [========>.....................] - ETA: 1s - loss: 0.1213 - accuracy: 0.9622
2048/5000 [===========>..................] - ETA: 1s - loss: 0.1229 - accuracy: 0.9639
2560/5000 [==============>...............] - ETA: 0s - loss: 0.1335 - accuracy: 0.9613
3072/5000 [=================>............] - ETA: 0s - loss: 0.1314 - accuracy: 0.9616
3584/5000 [====================>.........] - ETA: 0s - loss: 0.1279 - accuracy: 0.9621
4096/5000 [=======================>......] - ETA: 0s - loss: 0.1282 - accuracy: 0.9614
4608/5000 [==========================>...] - ETA: 0s - loss: 0.1354 - accuracy: 0.9609
5000/5000 [==============================] - 2s 419us/sample - loss: 0.1413 - accuracy: 0.9598
'''

# 학습 결과를 출력합니다.
Visulaize([('CNN', history)], 'loss')  # 아래 그래프


'''
# visual.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from elice_utils import EliceUtils
elice_utils = EliceUtils()

def Visulaize(histories, key='loss'):
    for name, history in histories:
        plt.plot(history.epoch, history.history[key], 
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])    
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")
'''
```
![image](https://user-images.githubusercontent.com/61646760/145599505-6deb67f8-7fe2-474d-838a-b522553ab0bc.png)
- keras를 활용하여 CNN 모델을 설정
  - 분류 모델에 맞게 마지막 레이어의 노드 수는 10개, `activation` 함수는 'softmax'로 설정

### MNIST 분류 CNN 모델 - 평가 및 예측
앞에 이어서 이번에는 CNN 모델을 평가하고 예측해 보자.
- Keras에서 CNN 모델의 평가 및 예측을 위해 필요한 함수
  - `model.evaluate(X, Y)`
    - 평가 방법
    - `evaluate()` 메서드는 학습된 모델을 바탕으로 입력한 feature 데이터 `X`와 label `Y`의 loss 값과 metrics 값을 출력한다.
  - `model.predict_classes(X)`
    - 예측 방법
    - `X` 데이터의 예측 label 값을 출력한다.

```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from visual import *
from plotter import *
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)


# MNIST 데이터 세트를 불러옵니다.
mnist = tf.keras.datasets.mnist

# MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    

# Train 데이터 5000개와 Test 데이터 1000개를 사용합니다.
train_images, train_labels = train_images[:5000], train_labels[:5000]
test_images, test_labels = test_images[:1000], test_labels[:1000]

# CNN 모델의 입력으로 사용할 수 있도록 (샘플개수, 가로픽셀, 세로픽셀, 1) 형태로 변환합니다.
train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)


# CNN 모델을 설정합니다.
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME', input_shape = (28,28,1)),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

# CNN 모델 구조를 출력합니다.
print(model.summary())

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 32)          9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                32832     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 52,298
Trainable params: 52,298
Non-trainable params: 0
_________________________________________________________________
None
'''

# CNN 모델의 학습 방법을 설정합니다.
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
              
# 학습을 수행합니다. 
history = model.fit(train_images, train_labels, epochs = 10, batch_size = 128, verbose = 2)

'''
Train on 5000 samples
Epoch 1/10
5000/5000 - 2s - loss: 7.0627 - accuracy: 0.4618
Epoch 2/10
5000/5000 - 2s - loss: 5.1598 - accuracy: 0.6436
Epoch 3/10
5000/5000 - 2s - loss: 5.0952 - accuracy: 0.6506
Epoch 4/10
5000/5000 - 2s - loss: 3.4510 - accuracy: 0.7048
Epoch 5/10
5000/5000 - 2s - loss: 0.3488 - accuracy: 0.8970
Epoch 6/10
5000/5000 - 2s - loss: 0.1482 - accuracy: 0.9552
Epoch 7/10
5000/5000 - 2s - loss: 0.0963 - accuracy: 0.9704
Epoch 8/10
5000/5000 - 2s - loss: 0.0609 - accuracy: 0.9840
Epoch 9/10
5000/5000 - 2s - loss: 0.0369 - accuracy: 0.9914
Epoch 10/10
5000/5000 - 2s - loss: 0.0250 - accuracy: 0.9948
'''

Visulaize([('CNN', history)], 'loss')  # visual 모듈 사용 (아래 참고)

'''
# visual.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from elice_utils import EliceUtils
elice_utils = EliceUtils()

def Visulaize(histories, key='loss'):
    for name, history in histories:
        plt.plot(history.epoch, history.history[key], 
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])    
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")
'''

"""
1. 평가용 데이터를 활용하여 모델을 평가합니다.
   loss와 accuracy를 계산하고 loss, test_acc에 저장합니다.
"""
loss, test_acc = model.evaluate(test_images, test_labels, verbose = 0)

"""
2. 평가용 데이터에 대한 예측 결과를 predictions에 저장합니다.
"""
predictions = model.predict_classes(test_images)

# 모델 평가 및 예측 결과를 출력합니다.
print('\nTest Loss : {:.4f} | Test Accuracy : {}'.format(loss, test_acc))
print('예측한 Test Data 클래스 :', predictions[:10])

'''
Test Loss : 0.1703 | Test Accuracy : 0.9490000009536743
예측한 Test Data 클래스 : [7 2 1 0 4 1 4 9 6 9]
'''

# 평가용 데이터에 대한 레이어 결과를 시각화합니다.
Plotter(test_images, model)  # plotter 모듈 사용 (아래 참고)

'''
레이어 이름 : conv2d
레이어 이름 : max_pooling2d
레이어 이름 : conv2d_1
레이어 이름 : max_pooling2d_1
레이어 이름 : conv2d_2
레이어 이름 : max_pooling2d_2
'''

'''
# plotter.py

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt

from elice_utils import EliceUtils
elice_utils = EliceUtils()

def Plotter(test_images, model):

    img_tensor = test_images[0]
    img_tensor = np.expand_dims(img_tensor, axis=0) 
    
    layer_outputs = [layer.output for layer in model.layers[:6]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)
    
    layer_names = []
    for layer in model.layers[:6]:
        layer_names.append(layer.name)
    
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
    
        size = layer_activation.shape[1]
    
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
    
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
            
                channel_image -= channel_image.mean() 
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255.).astype('uint8')
            
                display_grid[col * size : (col+1) * size, row * size : (row+1) * size] = channel_image
            
        scale = 1. / size
        print('레이어 이름 :', layer_name)
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig("plot.png")
        elice_utils.send_image("plot.png")
        
    plt.show()
'''
```
![image](https://user-images.githubusercontent.com/61646760/145671952-13a58069-efeb-4308-9565-abd67169d87d.png)
![image](https://user-images.githubusercontent.com/61646760/145672114-2524d068-d0f3-44de-ba06-b09e61bdc799.png)
