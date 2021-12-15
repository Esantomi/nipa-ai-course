# 타이타닉 생존자 예측하기
titanic 데이터를 학습하여 생존 여부를 예측하고자 합니다. 아래와 같이 데이터가 구성되어 있을 때, 의사결정나무를 활용하여 분류 알고리즘을 수행해 보겠습니다.

### titanic 데이터 구성

![image](https://user-images.githubusercontent.com/61646760/146216652-51020511-f0b6-43f1-bf74-ffef4dd1b7b8.png)

titanic 데이터는 `./data/titanic.csv` 경로에 저장되어 있으며, 결측값 처리 및 feature 엔지니어링 거쳐 학습용, 평가용 데이터로 분리되어 있습니다.

### 지시사항
1. `model` 객체에서 `fit` 메서드를 활용하여 학습용 데이터를 학습합니다.
2. 학습용 데이터의 accuracy 값과 평가용 데이터의 accuracy 값을 각각 `acc_train`, `acc_test`에 저장합니다.
