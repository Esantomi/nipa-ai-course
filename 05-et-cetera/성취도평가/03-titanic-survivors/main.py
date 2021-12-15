import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from elice_utils import EliceUtils

elice_utils = EliceUtils()


# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')

# 'PassengerId','Name','Ticket','Cabin' 열 삭제
titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis='columns')

# 결측값 처리
titanic.dropna(inplace=True)

# 카테고리 데이터와 숫자형 데이터 정리
cat_features = ['Sex', 'Embarked']
num_features=['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']

for variable in cat_features:
    
    # 더미 생성 함수 get_dummies를 사용하여 더미 데이터 열 생성
    dummies = pd.get_dummies(titanic[cat_features])
    
    # 더미 열과 기존 숫자형 데이터를 연결하여 df_x로 저장
    df_x = pd.concat([titanic[num_features], dummies],axis=1)

# label 데이터 분리
df_y = titanic['Survived']

# 학습용, 평가용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.25, random_state=42)

model = DecisionTreeClassifier(random_state=42)

"""
1. fit 메서드를 활용하여 학습용 데이터를 학습합니다.
"""
model.fit(X_train, y_train)


"""
2. 학습용 데이터의 accuracy 값과 평가용 데이터의 accuracy 값을 각각 acc_train, acc_test에 저장합니다.
"""
acc_train = model.score(X_train, y_train)
acc_test = model.score(X_test, y_test)

# 결과를 출력합니다.
print("학습용 데이터의 정확도 :", acc_train)
print("평가용 데이터의 정확도 :", acc_test)

'''
학습용 데이터의 정확도 : 0.9906367041198502
평가용 데이터의 정확도 : 0.7134831460674157
'''