# from elice_utils import EliceUtils
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
# elice_utils.send_image("plot.png")