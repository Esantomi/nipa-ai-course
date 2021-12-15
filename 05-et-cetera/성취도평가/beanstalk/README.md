# AI 실무 기본 성취도 평가
## 쑥쑥 자라라 콩나무야!
### “쑥쑥 자라라 콩나무야!”

잭은 여러 개의 신비한 힘을 지닌 콩을 심고 하늘까지 쑥쑥 자라는 콩나무의 모습을 보았어요.

쑥쑥 자란 콩나무들의 키, 둘레, 열린 콩의 갯수 데이터가 csv파일로 만들어져 있네요.

우리는 csv 데이터를 읽고, 정렬해서 어떤 나무가 가장 큰 나무인지 확인해 보고자 해요!

어떤 나무의 `height` 변수가 가장 큰 값을 갖는지 알아보고, 해당 나무의 정보를 출력하는 실습을 진행해 봐요.

![jack](https://user-images.githubusercontent.com/61646760/146124922-2a2fe612-cdfb-42c0-aa8a-5d6f89ca0743.png)

### 이렇게 해보세요!

- `height`가 가장 큰 나무의 정보를 출력하기 위해 나무들의 정보를 `height` 컬럼을 기준으로 내림차순 정렬하고 인덱싱이나 `head()` 함수를 이용하세요.

## 답안
```
import pandas as pd


# ./data/tree_data.csv 파일을 읽어서 작업해 보세요!
beanstalk = pd.read_csv("./data/tree_data.csv")
print(beanstalk.head(), "\n")

'''
   beans  circumference  height
0  72459           2.30  994.98
1  30853           1.08  671.04
2  80810           0.24  890.37
3  61073           2.89  309.05
4  13964           3.02  715.66
'''

"""
어떤 나무의 height 변수가 가장 큰 값을 갖는지 알아보고, 해당 나무의 정보를 출력하는 실습을 진행해 보자.

Tip : height가 가장 큰 나무의 정보를 출력하기 위해 나무들의 정보를 height 컬럼을 기준으로 내림차순 정렬하고 인덱싱이나 head()함수를 이용하세요.

"""

beanstalk = beanstalk.sort_values('height', ascending = False)  # height를 기준으로 내림차순 정렬
print(beanstalk.head(), "\n")

print(beanstalk.loc[0])
# print(beanstalk.iloc[0])
```
