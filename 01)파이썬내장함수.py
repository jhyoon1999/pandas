#집합
set([1,1,2,3,2,2,3,4,5,5])

a = {1,2,3,4,5}
b = {3,4,5,6,7,8}

a.union(b)
a.intersection(b)

#1. 리스트 내포 or 리스트 컴프리헨션
[i * 2 for i in range(10)]

#(1). 조건문과 함께
text = ['am', 'banana', 'car', 'dark', 'drink','effect']
[x.upper() for x in text if len(x) > 3]

#(2). 중첩 for 문
tuples = [(1,2,3), (4,5,6), (7,8,9)]
len(tuples)

simple_list = [x for tup in tuples for x in tup]
simple_list

#2. 순차형 데이터 타입에 활용 가능한 유용한 반복자 함수
#(1). enumerate()
items = [0,5,10,15,20]
for i, v in enumerate(items) :
    print(i, v)
    
#(2). zip
#여러 개의 순차형 데이터를 서로 짝지어서 반복할 필요가 있을때
items2 = [5,25,125, 256,512]
for i, v in zip(items, items2) :
    print(i,v)

#(3). sorted
#정렬되지 않은 데이터를 정렬된 순차형 데이터 타입으로 반환한다.
sorted('hello my friend!')

#3. map() 함수
#특정 함수와 정의된 리스트를 활용하여 새로운 리스트를 생성하는 역할을 한다.
#첫번째 매개변수에는 함수, 두번째 매개변수에는 리스트를 넣는다.
func1 = lambda x: x*3
list1 = list(range(2,12, 2))
list1
list(map(func1, list1)) #리스트의 각 요소에 해당 함수를 실행하는구나


