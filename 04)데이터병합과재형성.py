#%% 데이터 병합
import pandas as pd

#1. 위아래로 연결하기
#(1)
m1 = pd.Series(['파스타', '라멘', '냉면'], index = [1,2,3])
m2 = pd.Series(['돈가스', '피자', '치킨'], index = [4,5,6])
pd.concat([m1,m2])

#(2)
data_1 = pd.DataFrame({'음식명': ['돈가스', '피자', '초밥', '치킨', '탕수육'],
                    '카테고리': ['일식', '양식', '일식', '양식', '중식']})
data_1

data_2 = pd.DataFrame({'음식명': ['갈비탕', '냉면', '짜장면', '파스타', '라멘'],
                    '카테고리': ['한식', '한식', '중식', '양식', '일식']})
data_2

pd.concat([data_1, data_2])
pd.concat([data_1, data_2], ignore_index=True)

