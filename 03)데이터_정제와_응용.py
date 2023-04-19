import pandas as pd

titanic = pd.read_csv('datasets/titanic.csv')
titanic.head()
titanic.info()
titanic['PassengerId'] = titanic.PassengerId.astype('object')
titanic['Survived'] = titanic.Survived.astype('category')
titanic['Pclass'] = titanic.Pclass.astype('category')
titanic.shape
titanic.describe()
titanic.describe(exclude = ['float64', 'int64'])

#1. 조건식을 활용한 데이터 필터링
#(1). 단일조건
titanic[titanic.Pclass == 3].head()

#(2). 다중조건
titanic[(titanic.Pclass == 3) & (titanic.Sex == 'female')].head()

#(3). loc 조건부 필터링
titanic.loc[
    (titanic.Pclass == 3) & (titanic.Sex == 'female'),
    ['Name', 'Age', 'Fare']
    ]

#(4). 특정 값 포함 여부 필터링
titanic.Embarked.value_counts()
titanic[titanic.Embarked.isin(['S', 'C'])].head()

titanic[~titanic.Embarked.isin(['S', 'C'])].head()

#(5). 다중 컬럼에서 특정 값이 포함된 데이터 필터링하기
filter_male = titanic.Sex.isin(['male'])
filter_male

filter_pclass = titanic.Pclass.isin([1,2])
filter_pclass

titanic[filter_male & filter_pclass].head()

#2. 쿼리를 사용하여 데이터 필터링하기
#데이터를 필터링하는 방법 중 가장 추천하는 방법은 query() 함수를 사용하는 것이다.
#직관적이고 가독성이 높으며 간편하기 때문이다.
titanic.query('Pclass == [1,2] & Fare > 100')
titanic.query('Sex == ["female", "male"] & Pclass == 1')

#3. 결측값을 제외하고 데이터 필터링하기
#결측값이 아닌 데이터만 선택할 때는 notnull() 함수를 활용한다.
titanic.isna().sum()
titanic[titanic.Cabin.notnull()].isna().sum()

#4. 특정 문자가 포함된 데이터 필터링하기
#contains() 메서드는 문자열에만 사용할 수 있는 str 상위 모듈을 앞에 호출해야 사용할 수 있다.
titanic[titanic.Name.str.contains('Catherin')].Name.head()

#5. 다양한 기준으로 데이터 정렬하기
#(1). sort_values
titanic.sort_values(by=['Fare'], ascending = False).head(3)

#6. 결측값 처리하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv('datasets/titanic.csv')
titanic.isnull().sum()

plt.figure(figsize= (12,7))
sns.heatmap(titanic.isnull(), cbar = False)

titanic.info()

#(1). 결측값 삭제/제거
titanic.dropna().shape
titanic.dropna(how = 'any').shape
titanic.dropna(thresh = 2).shape
titanic.dropna(subset=['Age', 'Embarked']).shape

#(2). 결측값 대체/보간
titanic.Age.fillna(25)
titanic.Age.replace(to_replace = np.nan, value = 1000)

#평균
titanic.Age.fillna(titanic.Age.mean())

#전후값 : 앞에 있는 값을 참조하려면 ffill, 뒤에 있는 값을 참조하려면 bfill
titanic.Cabin
titanic.Cabin.fillna(method="ffill")

#보간법으로 결측값 채우기
titanic.Age
titanic.Age.interpolate(method='nearest', limit_direction = 'forward')

#7. 이상값 처리
def outlier_iqr(data, column) :
    global lower, upper
    
    q1 = np.quantile(data[column], 0.25)
    q3 = np.quantile(data[column], 0.75)
    iqr = q3 - q1
    
    cut_off = iqr * 1.5
    lower, upper = q1-cut_off, q3+cut_off
    data1 = data[data[column] > upper]
    data2 = data[data[column] < lower]
    
    print('총 이상값 개수는', data1.shape[0] + data2.shape[0], '이다.')
    
outlier_iqr(titanic, 'Fare')

plt.figure(figsize=(12,7))
sns.distplot(titanic.Fare, kde = False)
plt.axvspan(lower, titanic.Fare.min(), alpha = 0.2, color = 'red')
plt.axvspan(upper, titanic.Fare.max(), alpha = 0.2, color = 'red')
plt.show()

#8. 문자열 데이터 처리하기
#object보다는 string을 사용하자
titanic.drop(['PassengerId', 'Cabin'], axis = 1, inplace = True)
titanic.Name.dtype

titanic.Name = titanic.Name.astype('string')
titanic.Name.dtype

#(1). 문자열 분리하기
titanic.Name.str.split()

titanic.Name.str.split(pat = ",")

titanic.Name.str.split(expand = True)
titanic.Name.str.split(pat = ",", expand = True)

titanic['title'] =titanic.Name.str.split().str[1]
titanic.title.value_counts().head()
titanic.title.str.replace('.','', regex = False)

#(2). 포함여부 확인하기
titanic[titanic.Name.str.contains('Mr')].head(10)

#9. 람다를 활용한 데이터 처리
#map():단일 컬럼 즉, 시리즈에만 적용가능
#apply() : 단일 또는 복수 컬럼, 즉 데이터프레임과 시리즈 모두에 적용가능
#applymap() : 복수 컬럼, 즉 데이터프레임에만 적용 가능

#(1). 단일 컬럼에 람다 적용
titanic['Fare_int'] = titanic.Fare.map(lambda x: int(x + x /100))
titanic[['Fare', 'Fare_int']].head()

titanic['Fare_int'] = titanic.Fare.apply(lambda x: int(x + x /100))
titanic['Age'].apply(lambda x: 'Adult' if x >= 18 else 'Child')

#(2). 데이터프레임에 람다 적용
titanic[['SibSp', 'Parch']].apply(lambda x: x['SibSp'] + x['Parch'], axis = 1)
titanic[['SibSp', 'Parch']].apply(lambda x: x.sum(), axis = 1)

import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
list_columns = list(iris.columns)
list_columns

iris[list_columns[:4]].applymap(lambda x: x*10).head()
