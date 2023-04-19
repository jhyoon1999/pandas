#판다스 기본
import pandas as pd
import numpy as np

#1. 시리즈
ser = pd.Series([1,3,5,7,9])
ser

ser.index
ser.values

#(1). 넘파이 배열을 활용해서 시리즈 생성하기
data = np.random.randint(0,10,5)
data

index = ['a', 'b', 'c', 'd', 'e']
series = pd.Series(data = data, index = index, name = 'from_numpy')
series.name
print(series)

#(2). 딕셔너리를 활용해서 시리즈 생성하기 : key값을 인덱스로 활용한다.
data = {'a':1000, "b":2000}
series = pd.Series(data=data, name = 'from_dict')
series.name
print(series)

#(3). 시리즈를 리스트로 변환
series.tolist()

#2. 데이터프레임
import seaborn as sns

df = sns.load_dataset('penguins')
df.shape
df.info()
df.head()
df.tail()
df.size #모든 값의 개수
len(df)

#(1). 데이터 타입 확인하기
df.dtypes

#(2). 자동으로 데이터 타입 변경하기
df = df.convert_dtypes()
df.dtypes

#(3). 수동으로 데이터 타입 변경하기
df = df.astype({'species':'category'})
df.info()
df['island'] = df['island'].astype('category')
df.info()

#3. 기술통계 확인
import seaborn as sns
df = sns.load_dataset('penguins')

df.describe()
df.describe(include = 'all')
df.describe(include = [object])

#(1). 범파이로 백분위수 구하기
df = df.fillna(0)
np.percentile(df['bill_depth_mm'], q = [0,25,50,75,100])

#(2). 판다스로 백분위수 구하기
df.info()
df.quantile([0, .25, .5, .75, 1.0])

#(3). 데이터 수 파악하기
#count()함수는 각 컬럼이나 로우 기준으로 결측값이 아닌 모든 데이터의 셀 수를 계산한다.
df.count()

#4. 기술통계 시각화
import seaborn as sns

#(1). 막대그래프
sns.catplot(data= df, x='species', kind = 'count')

sns.barplot(data = df, x = 'species', y = 'bill_length_mm')

#(2). 히스토그램
sns.histplot(data = df, x = 'flipper_length_mm')

#5. 고윳값 확인
df['species'].unique()
df['species'].nunique()

df['species'].value_counts()
df['species'].value_counts(dropna = True)
df['species'].value_counts(normalize = True)
df.value_counts() #가능한 조합을 다 보여준다

#6. 동일한 데이터 타입의 컬럼만 선택하기
df.select_dtypes(include = ['float64']).columns
df.select_dtypes(exclude = ['object']).columns

#7.컬럼 이름 변경 : .rename

#8. 컬럼과 인덱스 교환하기
df[:11].transpose()

#9. 컬럼 호출하기
df.species

#10. 컬럼 순서 바꾸기
pd.Series(df.columns)
df.columns = df.columns[[0,1,6,2,3,4,5]]
df.head()

#11. 데이터 인덱싱
#(1). 문자형 인덱스 인덱싱하기
#인덱스가 문자형인 경우에는 loc 함수를 활용하여 인덱싱한다.
df = sns.load_dataset('penguins')
df = df[:11].transpose()
df.head()

df.loc['bill_length_mm']
df.loc[['bill_length_mm', 'sex']]

df.loc['bill_length_mm', 5]

#(2). 컬럼 기준으로 인덱싱하기
df = df.T
df.head()
df.loc[df['bill_length_mm'] > 40]
df[df['bill_length_mm'] > 40]

#(3). 위치 기반 인덱싱하기
#iloc를 이용한다.
df.iloc[[0]]
df.iloc[[0,2]]

#12. 인덱스 설정하기
df.set_index('species',inplace = True)
df.head()

df.reset_index(inplace = True)
df.head()
