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
w = [-2, 1, 1]

# AND Gate를 만족하는지 출력하여 확인
print('perceptron 출력')

for x in X:
    print('Input: ',x[0], x[1], ', Output: ',perceptron(w, x))

'''
perceptron 출력
Input:  0 0 , Output:  0
Input:  0 1 , Output:  0
Input:  1 0 , Output:  0
Input:  1 1 , Output:  1
'''
```

### 다층 퍼셉트론
- **다층 퍼셉트론(Multi-Layer Perceptron)**
  - 단층 퍼셉트론을 여러 개 쌓아 입력층과 출력층 사이에 은닉층(hidden layer)으로 추가한 것
    - 단층 퍼셉트론은 입력층과 출력층만 존재
    - **은닉층(hidden layer)**
      - 입력층과 출력층 사이의 모든 Layer
      - 은닉층이 많아지면, 깊은 신경망이라는 의미의 Deep Learning 단어 사용
  - 1986년 첫 번째 빙하기의 끝

## 02. 텐서플로우와 신경망

## 03. 다양한 신경망
