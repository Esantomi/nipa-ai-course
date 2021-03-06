# 인공지능은 어떻게 이미지를 인식하고 대처하는가?
인공지능은 어떻게 이미지를 인식할까요?

이번 실습에서는 숫자 손글씨 데이터를 학습한 인공지능 모델을 사용하여 실제 손글씨를 인식하는 과정을 살펴봅니다.

### 이미지 분할
손글씨를 인식하기 위해서 이미지에서 분석에 필요한 부분을 따로 분리해야 합니다.

아래의 이미지는 실제로 종이에 숫자를 적고 사진을 찍어 업로드한 사진입니다.

![image](https://user-images.githubusercontent.com/61646760/145756395-02a561c1-2b6a-4e2c-bd07-44e07f23412c.png)

위 이미지에서 숫자 부분을 분리하는 과정을 수행해 봅시다.

### 지시 사항
1. 아래 코드를 사용하여 손글씨 이미지에서 숫자 부분만 분리하는 과정을 수행해 보세요.
    - `ma.data_print()`
2. `실행` 버튼을 눌러 결과를 확인해 보세요.
3. `제출` 버튼을 눌러 숫자 부분이 올바르게 분리되었는지 확인해 보세요.

## 답안
```
import machine as ma

def main():
    
    """
    지시사항 1번. 손글씨 이미지에서 숫자 부분만 분리하는 과정을 수행해보세요.
    """
    ma.data_print()

if __name__ == "__main__":
    main()

```

### 실행 결과
```
원본 이미지를 출력합니다.
```
![image](https://user-images.githubusercontent.com/61646760/145756536-7076db31-774a-462d-b663-0357fbaa483b.png)
```
이미지를 분할할 영역을 표시합니다.
```
![image](https://user-images.githubusercontent.com/61646760/145756565-683213ea-8967-4e1d-b4a1-671c22b9490e.png)
```
분할 된 이미지를 출력합니다.
```
![image](https://user-images.githubusercontent.com/61646760/145756681-b6f95f06-9d3f-45c9-8824-b3f889660427.png)
![image](https://user-images.githubusercontent.com/61646760/145756695-dde692f1-08b3-464d-b446-96bb53709a6a.png)
![image](https://user-images.githubusercontent.com/61646760/145756702-b61490c3-9536-4350-be9a-afb2e982c130.png)
![image](https://user-images.githubusercontent.com/61646760/145756727-2293f9f0-4711-434f-89f4-8dfa109ffed3.png)
![image](https://user-images.githubusercontent.com/61646760/145756733-0c9fb1bb-395e-4295-a007-be738ce5e712.png)
![image](https://user-images.githubusercontent.com/61646760/145756749-4f2fec58-b799-4e74-828d-173aac62ba27.png)
![image](https://user-images.githubusercontent.com/61646760/145756755-5a27cf9a-de2b-48ba-a4c2-6c39e0e3c90d.png)
