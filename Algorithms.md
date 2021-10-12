## 트리(Tree)

### 트리는 가계도와 같은 계층적인 구조를 표현할 떄 사용하는 자료 구조

* 루트 노드: 부모가 없는 최상위 노드
* 단말 노드(leaf node): 자식이 없는 노드
* 크기(size): 트리에 포함된 모든 노드의 개수
* 깊이: 루트 노드로부터의 거리
* 높이: 깊이 중 최댓값
* 차수(degree): 각 노드의 (자식 방향) 간선 개수



#### > 기본적으로 트리의 크기가 n일 때, 전체 간선의 개수는 n-1개



## 이진 탐색 트리

### 이진 탐색이 동작할 수 있도록 고안된 자료 구조의 일종

#### 특징 : 왼쪽 자식 노드 < 부모 노드 < 오른쪽 자식노드



##### 데이터 조회 과정

1. 루트 노드 조회(왼쪽은 작은 값, 오른 쪽은 큰 값)
2. 현재 노드와 값을 비교 
3. 원소 탐색 완료 시 종료



## 트리의 순회(Tree Traversal)

### 트리 자료구조에 포함된 노드를 특정한 방법으로 한 번씩 방문하는 방법

#### 트리의 정보를 시각적으로 확인

### 대표적인 트리 순회 방법

- 전위 순회 : 루트를 먼저 방문
- 중위 순회 : 왼쪽 자식 후 루트 방문
- 후위 순회 : 오른쪽 자식을 방문한 뒤에 방문 



## 트리의 순회 구현 예제



class Node 

```python
class Node:
	def __init__(self, data, left_node, right_node):
        self.data = data
        self.left_node = left_node
        self.right_node = right_node

# 전위 순회
def pre_order(node):
    print(node.data, end = ' ')
    if node.left_node != None:
        pre_order(tree[node.left_node])
    if node.right_node != None:
        pre_order(tree[node.right_node])
        
# 중위 순회
def in_order(node):
    if node.left_node != None:
        in_order(tree[node.left_node])
    print(node.data, end=' ')
    if node.right_node != None:
        in_order(tree[node.right_node])
# 후위 순회
def post_order(node):
    if node.left_node != None:
        post_order(tree[node.left_node])
        if node.right_node != None:
            post_order(tree[node.right_node])
    print(node.data, end=" ")
    
n = int(input())
tree = {}

for i in range(n):
    data, left_node, right_node = input().split()
    if left_node == "None":
        left_node = None
    if right_node == "None":
        right_node = None
    tree[data] = Node(data, left_node, right_node)
    
pre_order(tree['A'])
print()
in_order(tree['A'])
print()
post_order(tree['A'])


```



## 데이터 업데이트가 가능한 상황에서의 구간 합 문제

### 바이너리 인덱스 트리

#### 2진법 인덱스 구조를 활용해 구간 합 문제를 효과적으로 해결해 줄 수 있는 자료구조를 의미

##### 펜윅트리라고도 함

### 



## 정렬 알고리즘

### 정렬이란 데이터를 특정한 기준에 따라 순서대로 나열하는 것을 의미



#### 선택 정렬

##### : 처리되지 않은데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 바꾸는 것을 반복



```python
array = [7,5,9,0,3,1,6,2,4,8]

for i in range(len(array)):
	min_index = i #가장 작은 원소의 인덱스
	for j in range(i+1, len(array)):
		if array[min_index] > array[j]:
			min_index = j
	array[i],array[min_index] = array[min_index], array[i]
print(array)
```



#### 선택 정렬의 시간 복잡도

- 선택 정렬은 n번 만큼 가장 작은 수를 찾아서 맨 앞으로 보내야 함

- 구현 방식에 따라서 사소한 오차는 있을 수 있지만, 전체 연산 횟수는 다음과 같음

- $$
  N + (N-1) + (N-2) +...+2
  = (N^2 + N - 2)/2
  = O(N^2)
  $$

  

#### 삽입 정렬

##### :처리되지 않은 데이터를 한씩 골라 적절한 위치에 삽입

##### - 구현 난이도는 높으나, 일반적으로 더 효율적으로 동작함



```python
array = [7,5,9,0,3,1,6,2,4,8]

for i in range(1, len(array)):
	for j in range(i, 0, -1): # 인덱스 i부터 1까지 1씩 감소하며 반복하는 문법
		if array[j] < array[j - 1] #한 칸씩 왼쪽으로 이동
			array[j], array[j - 1]  = array[j - 1], array[j]
		else: #자기보다 작은 데이터를 만나면 그 위치에서 멈춤
		break
print(array)
```



#### 삽입 정렬의 시간 복잡도

- O(N^2), 선택정렬과 마찬가지로 반복문이 두 번 중첩
- 현재 리스트의 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작



#### 퀵 정렬

- 기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
- 일반적인 상황에서 가장 많이 사용되는 정렬 알고리즘 중 하나
- 병합 정렬과 더물어 대부분의 프로그래밍 언어의 정렬 라이브러리의 근간이 되는 알고리즘
- 첫 번째 데이터를 기준 데이터(Pivot)로 설정



```
array = [7,5,9,0,3,1,6,2,4,8]

def quick_sort(array, start, end):
	if start >= end: # 원소가 1개인 경우 종료
		return
	pivot =  start #피벗은 첫 번째 원소
	left = start + 1
	right = end
	while (left <= right):
		#피벗보다 큰 데이터를 찾을 때까지 반복
		while(left <= end and array[left] <= array[pivot]):
			left += 1
		#피벗보다 작은 데이터를 찾을 때까지 반복 
		while(right > start and array[right] >= array[pivot]):
			right -= 1
		if(left > right): #엇갈렸다면 작은 데이터와 피벗을 교체
			array[right], array[pivot] = array[pivot], array[right]
		else: #엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
		array[left], array[right] = array[right], array[left]
	#분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
	quick_sort(array, start, right-1)
	quick_sort(array, right+1, end)
	
quick_sort(array, 0, len(array) - 1)
print(array)
```



##### 퀵 정렬의 시간 복잡도

- 평균 O(NlogN)의 시간 복잡도
- 최악의 경우 O(N^2)



### 퀵 정렬 소스코드: 파이썬의 장점

```python
array = [7,5,9,0,3,1,6,2,4,8]

def quick_sort(array):
	# 리스트가 하나 이하의 원소만을 담고 있다면 종료
	if len(array) <= 1:
		return array
	pivot = array[0] # 피벗은 첫 번째 원소
	tail = array[1:] # 피벗을 제외한 리스트
	
	left_side = [x for x in tail if x <= pivot]#분할된 왼쪽 부분
	right_side = [x for x in tail if x > pivot]#분할된 오른쪽 부분
	
	#분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행하고, 전체 리스트 반환
	return quick_sort(left_side) + [pivot] + quick_sort(right_side)
	
pring(quick_sort(array))
```



#### 계수 정렬

- 특정한 조건이 부합할 때만 사용가능, 속도가 매우 빠름
  - 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때 사용
- 데이터의 개수가 N, 데이터(양수) 중 최댓값이 K일 때 최악의 경우에도 수행 시간 O(N+K) 보장



```python
# 모든 원소의 값이 0보다 크거나 같다고 가정
array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
# 모든 범위를 포함하는 리스트 선언(모든 값은 0으로 초기화)
count = [0] * (max(array) + 1)

for i in range(len(array)):
	count[array[i]] += 1 #각 데이터에 해당하는 인덱스의 값 증가
	
for i in range(len(count)): # 리스트에 기록된 정렬 정보 확인
	for j in range(count[i]):
		print(i, end = ' ') # 띄어쓰기를 구분으로 등장한 횟수만큼 인덱스 출력
		

	
```



####  계수 정렬의 복잡도

- 시간복잡도와 공간복잡도는 O(N+K)
- 때에 따라 심각한 비효율성
- 동일한 값을 가지는 데이터가 여러 개 등장할 때 효과적
  - 예) 성적 등