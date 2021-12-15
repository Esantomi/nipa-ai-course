# 토끼와 거북이 경주 결과
### “나랑 달리기 시합하지 않을래?”

토끼와 거북이가 달리기 시합을 하기로 했어요.
공정한 경쟁을 위해서 1초마다 토끼와 거북이의 위치를 다른 동물이 기록하기로 하고 경주를 했네요.

그 위치 데이터가 `csv` 파일로 저장되어 있어요. 우리는 `csv` 파일을 읽어서 토끼와 거북이의 시간별 위치를 그래프로 시각화해 보고자 합니다!

![hare](https://user-images.githubusercontent.com/61646760/146125685-c237b122-ff5e-4632-9d03-e4fabb851db2.png)

### 이렇게 해보세요!

1. `race.csv` 파일을 읽어와 `df` 변수에 저장하고 시간 컬럼을 `index`로 지정해 보세요.
2. 토끼와 거북이 컬럼을 각각 `plot()` 함수를 통해 하나의 그래프에 그리고 `label`을 각 컬럼의 이름으로 설정해 보세요.
3. 범례의 위치를 오른쪽 아래의 위치로 설정해 보세요.

## 답안
```
from elice_utils import EliceUtils
from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = 'NanumBarunGothic'

elice_utils = EliceUtils()


# 아래 경로에서 csv 파일을 읽어서 시각화해 보세요.
# 경로: "./data/race.csv"
df = pd.read_csv("./data/race.csv")
print(df.head(), "\n")

'''
   시간  토끼  거북이
0   0   0    0
1   1   9    3
2   2  20    8
3   3  31   12
4   4  42   16
'''

df = df.set_index("시간")  # 시간을 df의 index로 설정
print(df.head(), "\n")

'''
    토끼  거북이
시간         
0    0    0
1    9    3
2   20    8
3   31   12
4   42   16
'''

fig, ax = plt.subplots()
# 토끼 컬럼과 거북이 컬럼을 label과 함께 plot()함수로 그려 보세요.
ax.plot(df['토끼'], label = "토끼")
ax.plot(df['거북이'], label = "거북이")

# 범례의 위치를 오른쪽 아래로 설정해 보세요.
ax.legend(loc=4)  # 위치 옵션 4 (lower right)

# elice에서 그래프 확인하기
fig.savefig("plot.png")
elice_utils.send_image("plot.png")
```

![image](https://user-images.githubusercontent.com/61646760/146125987-63f074ea-d325-4395-8752-a1c3adb26564.png)
