# 분할된 이미지의 숫자 예측
첫 번째 실습을 통하여 예측할 이미지를 분할하였고, 두 번째 실습을 통하여 이를 예측할 모델을 구현하였습니다.

예측을 위해서는 분할된 이미지를 MNIST와 비슷한 형태로 변환하는 과정이 필요합니다.

실제 분할된 이미지는 크기도 제각각에 컬러가 있는 이미지이기에 MNIST 데이터 형태와 동일하게 사이즈 조정 및 흑백 변환을 수행합니다.

이후 최종적으로 손글씨 분류 모델을 활용하여 손글씨 숫자를 인식하여 봅시다.

### 지시 사항
1. 아래 코드를 사용하여 이미지 변환 과정과 예측 과정을 수행해 보세요.
    - `ma.data_predit()`
2. `실행` 버튼을 눌러 결과를 확인해 보세요.
3. `제출` 버튼을 눌러 이밎 변환 및 예측을 올바르게 수행했는지 확인해 보세요.

## 답안
```
import machine as ma

def main():
    
    """
    지시사항 1번. 이미지 변환 과정과 예측 과정을 수행해 보세요.
    """
    ma.data_predit()

if __name__ == "__main__":
    main()
```

### 실행 결과
![image](https://user-images.githubusercontent.com/61646760/145757722-3656bdda-100d-42d8-8563-8312347203fd.png)
```
Prediction:  2
```
![image](https://user-images.githubusercontent.com/61646760/145757731-bcdf827b-306f-4616-823d-0193b6ad9f14.png)
```
Prediction:  8
```
![image](https://user-images.githubusercontent.com/61646760/145757748-25b58ef4-9be8-4667-a525-1e843ffde4ba.png)
```
Prediction:  4
```
![image](https://user-images.githubusercontent.com/61646760/145757766-57e77930-862a-4b6c-afa4-76c60fc1e884.png)
```
Prediction:  3
```
![image](https://user-images.githubusercontent.com/61646760/145757785-cbcef7e6-6e49-4e6c-a9db-d155f0002430.png)
```
Prediction:  9
```
![image](https://user-images.githubusercontent.com/61646760/145757804-a2cb9e5d-9ee1-48c3-b5d5-530d82edcf56.png)
```
Prediction:  5
```
![image](https://user-images.githubusercontent.com/61646760/145757881-0036b455-8320-4872-9c5e-68aedf023fd9.png)
```
Prediction:  8
```
