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