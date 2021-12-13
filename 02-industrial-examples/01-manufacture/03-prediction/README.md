# 공정 이상 예측하기
이전 실습의 학습 결과를 통하여 103번 센서의 관측 값이 중요하다는 것을 알아보았습니다.

![image](https://user-images.githubusercontent.com/61646760/145755332-5a8973ac-683c-4ffa-9052-559faa9649bf.png)

이번 실습에서는 103번 센서의 값을 조절 했을 때 공정 이상이 발생할 지를 학습된 인공지능 모델을 사용하여 예측하여 봅시다.

아래는 103번 센서의 관측치 값들의 분포를 나타낸 그래프입니다.

![image](https://user-images.githubusercontent.com/61646760/145755349-dac4966e-4e99-49ce-88a2-d987268c9a5b.png)

### 지시 사항
1. None을 지우고 103번 센서의 관측치 값을 직접 입력하고자 합니다. `value_103_sensor`에 정상 범위의 값 `-0.02이상 0이하`를 입력하세요.
2. `실행` 버튼을 눌러 결과를 확인해 보세요.
3. `제출` 버튼을 눌러 정상 범위의 값을 올바르게 입력했는지 확인해 보세요.

### Tips
정상 범위 밖의 값을 입력한 후 실행 버튼을 눌러 결과를 확인해 보세요.

## 답안
```
import machine as ma

def main():
	
    """
    지시사항 1번. 103번 센서값인 아래의 value_103_sensor 값을 바꾸어 보세요.
    """
    
    # 103번 센서 값인 아래의 value_103_sensor 값을 바꾸어 보며 <실행> 버튼으로 확인해 보자.
    value_103_sensor = -0.01
    
    # 예측을 진행하는 코드입니다.
    ma.predict(value_103_sensor)
    
if __name__ == "__main__":
    main()
```

### 실행 결과
![image](https://user-images.githubusercontent.com/61646760/145755517-4fc55264-23ff-4dfc-afa9-78695e0cd96f.png)
```
103 센서 데이터 값이 -0.01인 경우 공정 이상이 발생하지 않을 것으로 예측됩니다.
```
