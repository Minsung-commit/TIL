#!/usr/bin/env python
# coding: utf-8

# # 분류(classification)

# **지도학습** 
# - 레이블(정답)이 있는 데이터가 주어진 상태에서 학습하는 머신러닝 방식
# 
# 
# **분류(Classification)**
# - 지도학습의 대표적 유형
# - 학습 데이터로 주어진 데이터의 피처와 레이블값(결정값, 클래스값)을 머신러닝 알고리즘으로 학습해 모델을 생성하고 생성된 모델에 새로운 값이 주어졌을 때 미지의 레이블 값을 예측하는 것
# - 기존 데이터가 어떤 레이블에 속하는지 알고리즘을 통해 패턴을 인지한 뒤 새롭게 돤측된 데이터에 대한 레이블을 판별

# ### 대표적인 분류 알고리즘
# - 결정 트리(Decision Tree) : 데이터 균일도에 따른 규칙 기반
# - 나이브 베이즈(Navie Bayes) : 베이즈 통계와 생성 모델에 기반
# - 로지스틱 회귀(Logistic Regression) : 독립변수와 종속변수의 선형 관계성에 기반
# - 서포트 벡터 머신(Support Vector Machine) : 개별 클래스 간의 최대 분류 마진을 효과적으로 찾음
# - 최소 근접 알고리즘(Nearest Neighbor) : 근접 거리 기준
# - 신경망(Neural Network) : 심층 연결 기반
# - 앙상블(Ensemble) : 여러 머신러닝 알고리즘 결합

# ## 결정 트리(Decision Tree) 
# - 학습을 통해 데이터에 있는 규칙을 자동으로 찾아내 트리 기반의 분류 규칙을 만드는 알고리즘
# - ML 알고리즘 중 직관적으로 이해하기 쉬운 알고리즘으로 분류와 회귀 문제에서 가장 널리 사용
# - 특정기준(질문)에 따라서 데이터를 구분
# - 가장 쉬운 규칙 표현 방법 : if/else 기반 (스무고개 게임과 유사)
# 
# ### 의사결정나무의 타입
# - 분류 나무
#     - 범주형 목표 변수를 기준으로 마디를 나눔
#     - 끝마디에 포함된 자료의 범주가 분류 결과 값이 됨
# 
# - 회귀 나무
#     - 연속형 목표 변수를 기준으로 마디를 나눔
#     - 끝마디에 포함된 자료의 평균값이 각 끝마디의 회귀 값이 됨  
# 
# ### Tree 구조
# - 전체적인 모양이 나무를 뒤집어 놓은 것과 닮았다고 해서 붙여진 이름
# - 결정트리에서 질문이나 네모상자를 노드(Node) 라고 함
# - 맨 위의 노드(첫 질문)를 Root Node
# - 각 질문에 대한 마지막 노드를 Leaf Node : 결정된 클래스 값 
# - Decision Node(규칙 노드) : 규칙 조건  
# - 새로운 규칙 조건마다 Sub Tree 생성 
# 
# 
# ![image.png](attachment:image.png)

# ### 결정트리에서 중요한 이슈
# - 데이터 세트의 피처가 결합해 규칙 조건을 만들 때마다 규칙노드가 만들어짐  
# - `규칙이 많아지면` 결정 방식이 복잡해지고 `과적합(overfitting)` 발생  
#     - 즉, `depth가 길어질수록` 결정트리의 예측 성능이 저하될 가능성이 높음  
#     
#     
# - 가능한 적은 결정노드로 높은 예측 정확도를 가지려면  
#     - 데이터를 분류할 때 최대한 많은 데이터 세트가 해당 분류에 속할 수 있도록 결정 노드의 규칙이 정해져야 함  
# 
# 
# - **`어떻게 트리를 분할할 것인가`가 중요**  
#     - 최대한 균일한 데이터 세트를 구성할 수 있도록 분할하는 것이 필요  

# ### 가지치기 (pruning)
# - 특정 노드 밑의 하부 트리를 제거하여 일반화 성능을 높이는 방법
# - 깊이가 줄어들고 결과의 개수가 줄어듦
# - 트리에 가지가 너무 많으면 과적합(overfitting)
# - 과적합(overfitting)을 막기 위한 방법

# ### 결정트리의 장단점
# 
# **장점**
# - 매우 쉽고 유연하게 적용될 수 있는 알고리즘
# - 정보 균일도 룰을 기반으로 알고리즘이 쉽고 직관적이며 명확함
# - 데이터 스케일링이나 정규화 등의 전처리 영향이 적음
# 
# **단점**
# - 규칙이 많아지면 결정 방식이 복잡해지고 과적합(overfit) 발생
# - 즉, depth가 길어질수록 결정트리의 예측 성능이 저하될 가능성이 높음 

# ### 결정트리 알고리즘 성능
# - **`데이터의 어떤 기준을 바탕으로`** 규칙을 만들어야 `가장 효율적인 분류`가 될 것인가가 `알고리즘의 성능`을 크게 좌우
# - 가능한 적은 결정노드로 높은 예측 정확도를 가지려면 
#     - 데이터를 분류할 때 최대한 많은 데이터 세트가 해당 분류에 속할 수 있도록 결정 노드의 규칙이 정해져야 함
# - 그러기 위해서는 **`어떻게 트리를 분할할 것인가가`** 중요
# - 최대한 균일한 데이터 세트를 구성할 수 있도록 분할하는 것이 필요

#  ### 결정 트리 구조

# ![image.png](attachment:image.png)

# ## 균일도(불순도: impurity) 
# 
# - 규칙 조건 생성에서 중요한 것
# - 정보 균일도가 높은 데이터 세트를 먼저 선택할 수 있도록 규칙 조건을 만드는 것이 중요
# 
# **균일한 데이터 세트**
# - 데이터 세트의 균일도는 데이터를 구분하는 데 필요한 정보의 양에 영향을 미침

# ![image.png](attachment:image.png)

# A : 하얀공, 검은공 유사하게 섞여 있음. 혼잡도가 높고 균일도가 낮음  
# B : 일부 하얀공, 대부분 검은 공  
# C : 모두 검은공. 가장 균일도 높음  
# 
# C에서 데이터를 추출했을 때 데이터에 대한 별다른 정보 없이도 '검은공'이라고 쉽게 예측 가능
# 
# A의 경우는 상대적으로 혼잡도가 높고 균일도가 낮기 때문에 같은 조건에서 데이터를 판단하는 데 더 많은 정보 필요

# ### 엔트로피(Entropy)
# - 데이터 분포의 불순도(impurity)를 수치화한 지표
# - 서로 다른 데이터가 섞여 있으면 엔트로피가 높고 같은 값이 섞여 있으면 엔트로피 낮음
# - 엔트로피를 통해 불순도 판단
#     - 수치 0: 모든 값이 동일 (분류하지 않아도 됨)
#     - 수치 1: 불순도 최대
# 
# ### 정보 이득 (Information Gain) 지수
# - 분류를 통해 정보에 대해 얻은 이득
# - **`1 - 엔트로피 지수`**
# - 결정 트리는 정보 이득 지수로 분할 기준을 정함
# - 즉, 정보 이득 지수가 높은 속성을 기준으로 분할
# 
# ### 지니(Gini) 계수
# - 불순도(impurity)를 수치화한 지표
#     - 경제학에서 불평등 지수를 나타낼 때 사용하는 계수
#     - 0이 가장 평등하고, 1로 갈수록 불평등
# - 머신러닝에 적용될 때는 지니 계수가 낮을수록 데이터 균일도가 높은 것으로 해석
# - 지니 계수가 낮은 속성을 기준으로 분할
# 
# ### 결정 트리 알고리즘에서 지니 계수 이용
# - 사이킷런의 **`DecisionTreeClassifier 클래스`**는 
# - 기본으로 **`지니 계수를 이용`**해서 데이터 세트 분할
# - 데이터 세트를 분할하는 데 가장 좋은 조건
#     - 정보 이득이 높거나 지니 계수가 낮은 조건

# ![image.png](attachment:image.png)

# 

# ### 결정 트리 알고리즘에서 분류를 결정하는 과정
# - 정보 이득이 높거나 지니 계수가 낮은 조건을 찾아서
# - 자식 트리 노드에 걸쳐 반복적으로 분할한 뒤
# - 데이터가 모두 특정 분류에 속하게 되면 분할을 멈추고 분류 결정

# **[분류 결정 과정]**

# ![image.png](attachment:image.png)

# ## 사이킷런의 결정트리 알고리즘 클래스
# 
# - 분류모델을 위한 **`DecisionTreeClassifier`**
# - 회귀모델을 위한 **`DecisionTreeRegressor`**
# - 분류와 회귀모델 모두 CART(Classification And Regression Tree) 알고리즘 기반
# 
# - 결정 트리를 위한 클래스의 매개변수
#     - 분류, 회귀 모두 동일하게 적용
#     
# ![image.png](attachment:image.png)

# ## 결정 트리 모델의 시각화(Decision Tree Visualization)

# ### Graphviz 패키지 사용
# - 사이킷런에서 export_graphviz() API 제공
# - 학습된 결정 트리 규칙을 실제 트리 형태로 시각화

# ### Graphviz 설치  
# 1. Graphviz 파이선 래퍼 모듈 설치 (윈도우버전)
#     - https://graphviz.org/download/
#     - graphviz-2.**.*.msi 다운로드
#     
# 
# 2. Graphviz 설치 후 Graphviz 파이썬 래퍼 모듈을 pip로 설치
#     - pip install graphviz
#     - (아나콘다 Command 콘솔 생성 시 관리자권한으로 실행)
# ![image-3.png](attachment:image-3.png)
#     
# 
# 3. 환경변수 설정
#     - Graphviz가 설치된 경로 : C:\Program Files\Graphviz\bin
#     - '내 PC'이 속성 -> 고급시스템설정 -> 환경변수 클릭
#     - Path와 시스템 변수 Path에 경로 추가
#     
#     
# 4. 주피터노트북 서버 프로그램 다시 실행
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# ### 붓꽃 데이터 세트에 결정 트리 적용 및 시각화
# - DecisionTreeClassifier 이용해 학습한 뒤 
# - 규칙 트리 시각화

# In[1]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# DecisionTree Classifier 생성
dt_clf = DecisionTreeClassifier(random_state=156)

# 붓꽃 데이터를 로딩하고, 학습과 테스트 데이터 셋으로 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,
                                                       test_size=0.2,  random_state=11)

# DecisionTreeClassifer 학습. 
dt_clf.fit(X_train , y_train)


# ### sklearn.tree모듈은 Graphviz를 이용하기 위한 export_graphviz()함수 제공
# 
# **export_graphviz() 함수**
# 
# 매개변수
# - 학습이 완료된 estimator
# - output 파일명
# - 결정 클래스 명칭
# - 피처 이름
# - impurity=True : 각 노드에 불순도 표시 (gini 계수) (True: 디폴트)
# - filled=True : 노드 색상 표시 (False: 디폴트)    

# In[2]:


from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
export_graphviz(dt_clf, out_file='tree.dot', class_names=iris_data.target_names,
               feature_names=iris_data.feature_names, impurity=True, filled=True)


# 생성된 .dot 파일 출력 방법 2가지
# 1. Graphviz 시각화툴 사용 : .dot 파일 읽어서 출력
# 2. 이미지 파일로 변환해서 저장 후 출력


# In[3]:


# 1. Graphviz 시각화툴 사용 : .dot 파일 읽어서 출력
import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화 

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# ---

# In[4]:


# 2. 생성된 .dot 파일을 .png 파일로 변환해서 저장 후 출력
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=600'])

# jupyter notebook에서 .png 직접 출력
from IPython.display import Image
Image(filename = 'decistion-tree.png')


# In[5]:


##### 참고 : 
from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names , feature_names = iris_data.feature_names, impurity=False, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# ### 생성된 트리 설명
# - 리프 노드 : 최종 클래스(레이블) 값이 결정되는 노드
# - class = setosa (레이블 값)
# 
# 리프 노드가 되기 위한 조건
# - 하나의 클래스 값으로 최종 데이터가 구성되거나
# - 리프 노드가 될 수 있는 하이퍼 파라미터 조건 중족하면 됨
# 
# 브랜치 노드
# - 자식 노드가 있는 노드
# - 자식 노드를 만들기 위한 분할 규칙 조건을 가지고 있음
# 
# 노드 내에 기술된 지표의 의미
# - 분할 규칙 조건 : petal length (cm) <= 2.45
#     - 리프 노드에는 조건이 없음
# - gini 계수 : gini = 0.667 (불순도)
#     - 리프 노드 : gini = 0.0 
#     - 결정트리는 불순도를 최소화하는게 목표
# - 현 규칙에 해당하는 데이터 건수 : samples = 120
# - 클래스 값 기반의 데이터 건수 : value = [41, 40, 39]
#     - Setosa 41개, Versicolor 40개, Virginica 39개
# - 최종 클래스(레이블) 값 : class = setosa
#     
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# 2번 노드 (리프 노드)
# - 모든 데이터가 Setosa로 결정되므로
# - 더 이상 노드에 규칙을 만들 필요 없음
# - 지니 계수는 0
# 
# ![image.png](attachment:image.png)

# 3번 노드
# - petal length (cm) <= 2.45가 False인 규칙 노드
# - 79개의 샘플 데이터 중 Versicolor 40개, Virginica 39개로 
# - 지니 계수는 0.5로 높으므로
# - 다음 자식 브랜치 노드로 분기할 규칙 필요
# - petal width (cm) <= 1.55 규칙으로 자식 노드 생성
# 
# ![image.png](attachment:image.png)

# 4번 노드 
# - petal width (cm) <= 1.55가 True인 규칙 노드
# - 38개의 샘플 데이터 중 Versicolor 37개 Virginica 1개로 대부분 Verisicolor임
# - 지니 계수는 0.051로 매우 낮으나
# - 여전히 Virsicolor와 Virginica가 섞여 있으므로
# - petal length (cm) <= 5.25라는 새로운 규칙으로
# - 다시 자식 노드 생성
# 
# ![image.png](attachment:image.png)

# 각 노드의 색상은 붓꽃 데이터의 레이블 값을 의미
# - 주황색 : 0 Setosa
# - 초록색 : 1 Versicolor
# - 보라색 : 2 Virginica
# - 색상이 짙어질수록 지니 계수가 낮고 
# - 해당 레이블에 속하는 샘플 데이터가 많다는 의미

# ## 결정 트리 하이퍼 파라미터
# - 규칙 트리는 규칙 생성 로직을 미리 제어하지 않으면
# - 완벽하게 클래스 값을 구별해 내기 위해 
# - 트리 노드를 계속해서 만들어가기 때문에
# - 매우 복잡한 규칙 트리가 만들어져
# - 모델이 쉽게 과적합되는 문제 발생
# - 하이퍼 파라미터를 사용하여 복잡한 트리가 생성되지 않도록 제어

# ### 결정 트리의 하이퍼 파라미터
# - max_depth : 결정 트리의 최대 트리 깊이 제어
# - min_samples_split : 자식 규칙 노드를 분할해서 만드는데 필요한 최소 샘플 데이터 개수
#     - min_samples_split=4로 설정하는 경우 
#     - 최소 샘플 개수가 4개 필요한데
#     - 3개만 있는 경우에는 더 이상 자식 규칙 노드를 위한 분할을 하지 않음
#     - 트리 깊이도 줄어서 더 간결한 결정 트리 생성
# - min_samples_leaf : 리프 노드가 될 수 있는 최소 샘플 개수
#     - 리프 노드가 될 수 있는 조건은 디폴트로 1
#     - 즉, 다른 클래스 값이 하나도 없이 단독 클래스로만 되어 있거나
#     - 단 한 개의 데이터로 되어 있을 경우에 리프 노드가 될 수 있다는 것
#     - min_samples_leaf 값을 키우면 더 이상 분할하지 않고, 
#     - 리프 노드가 될 수 있는 조건이 완화됨
#     - min_samples_leaf=4로 설정하면 
#     - 샘플이 4 이하이면 리프 노드가 되기 때문에
#     - 지니 계수가 크더라도 샘플이 4인 조건으로 규칙 변경되어
#     - 브랜치 노드가 줄어들고 결정 트리가 더 간결하게 됨

# In[6]:


# 결정 트리 하이퍼 파라미터 튜닝
# max_depth = 3으로 조정
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=156)
dt_clf.fit(X_train , y_train)

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names,
                feature_names = iris_data.feature_names, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# 트리 형태 간결하게 됨


# In[7]:


# 결정 트리 하이퍼 파라미터 튜닝
# max_depth = 4으로 조정
dt_clf = DecisionTreeClassifier(max_depth=4, random_state=156)
dt_clf.fit(X_train , y_train)

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names,
                feature_names = iris_data.feature_names, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# 트리 형태 간결하게 됨


# In[8]:


# 결정 트리 하이퍼 파라미터 튜닝
# min_samples_split=4로 상향 
dt_clf = DecisionTreeClassifier(min_samples_split=4, random_state=156)
dt_clf.fit(X_train , y_train)

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, 
                feature_names = iris_data.feature_names, impurity=True, filled=True)


with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# 샘플 개수가 3개인데도 더 이상 자식 규칙 노드를 위한 분할을 하지 않음
# 트리 깊이도 줄어서 더 간결한 결정 트리 생성


# In[9]:


# 결정 트리 하이퍼 파라미터 튜닝
# min_samples_split=4로 상향 
dt_clf = DecisionTreeClassifier(min_samples_split=5, random_state=156)
dt_clf.fit(X_train , y_train)

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, 
                feature_names = iris_data.feature_names, impurity=True, filled=True)


with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# 샘플 개수가 3개인데도 더 이상 자식 규칙 노드를 위한 분할을 하지 않음
# 트리 깊이도 줄어서 더 간결한 결정 트리 생성


# In[10]:


# 결정 트리 하이퍼 파라미터 튜닝
# min_samples_leaf=4로 상향 
dt_clf = DecisionTreeClassifier(min_samples_leaf=4, random_state=156)
dt_clf.fit(X_train , y_train)

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names , feature_names = iris_data.feature_names, impurity=True, filled=True)


with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# 샘플이 4 이하이면 리프 노드가 되기 때문에
# 지니 계수가 크더라도 샘플이 4인 조건으로 규칙 변경되어
# 브랜치 노드가 줄어들고 결정 트리가 더 간결하게 됨


# ### feature_importances_ 속성 (피처 중요도)
# - tree를 만드는 결정에 각 피처가 얼마나 중요한지 평가
# - 0과 1사이의 숫자
# 
# 결정 트리는 균일도에 기반해서 어떠한 속성(피처)을 규칙 조건으로 선택하느냐가 중요한 요건  
# 중요한 몇 개의 피처가 명확한 규칙 트리를 만드는데 크게 기여하며  
# 모델을 좀 더 간결하고 이상치에 강한 모델을 만들 수 있음
# 
# **`feature_importances_`** 
# - ndarray 형태로 값을 반환
# - 피처 순서대로 값이 할당됨
# - [첫 번째 피처의 중요도, 두 번째 피처의 중요도, ....]
# - 값이 높을 수록 해당 피처의 중요도가 높다는 의미
# - 특정 노드의 중요도 값이 클수록, 그 노드에서 불순도가 크게 감소 의미

# In[11]:


# 피처별 중요도 값 확인하고 막대 그래프로 표현
# 위 예제에서 fit()으로 학습된 DecisionTreeClassifier 객체 df_clf의 
# feature_importances_ 속성 사용

import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# 피처 이름
print('iris_data.feature_names:', iris_data.feature_names)

# feature importance 추출 
print('Feature Importances : \n{0}'.format(np.round(dt_clf.feature_importances_, 3)))


# In[12]:


# feature별 importance 매핑
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print('{0}: {1:.3f}'.format(name, value))


# In[13]:


# feature importance를 column 별로 시각화 하기 
sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)

# 결과
# petal length의 피처 중요도가 가장 높음


# In[14]:


# feature_names을 X축으로 변경

sns.barplot(x=iris_data.feature_names, y=dt_clf.feature_importances_)


# ## 결정 트리(Decision TREE) 과적합(Overfitting)

# * 과적합(overfitting) : ML에서 학습 데이타를 과하게 학습하는 것  
# * 과적합 문제 : 학습(training) 데이터에서는 정확하지만 테스트 데이터에서는 성과가 나쁜 현상  
# * 의사결정 트리 과적합 : 나무의 크기가 크거나 가지 수가 많을 때 과적합 문제 발생  

# ### 결정 트리 과적합 시각화
# - 사이키럿의 make_classification() 함수 사용 
# - 결정 트리가 어떻게 학습 데이터를 분할하고 예측을 수행하는지,
# - 이로 인한 과적합 문제를 시각화
# - 2개의 피처가 3가지 유형의 클래스 값을 가지는 데이터 세트 생성하고 시각화

# ### `make_classification()` 함수
# - 분류용 가상 데이터를 생성
# 
# 인수
# - n_samples : 표본 데이터의 수, 디폴트 100
# - n_features : 독립 변수의 수, 디폴트 20
# - n_informative : 독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수, 디폴트 2
# - n_redundant : 독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수, 디폴트 2
# - n_repeated : 독립 변수 중 단순 중복된 성분의 수, 디폴트 0
# - n_classes : 종속 변수의 클래스 수, 디폴트 2
# - n_clusters_per_class : 클래스 당 클러스터의 수, 디폴트 2
# - weights : 각 클래스에 할당된 표본 수
# - random_state : 난수 발생 시드
#     
# 반환값
# - 독립 변수 X : (n_samples, n_features) 크기의 배열
# - 종속 변수 y : (n_samples,) 크기의 배열

# In[16]:


# 분류용 가상 데이터 생성하고 시각화

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 2차원 시각화를 위해서 feature는 2개, 결정값 클래스는 3가지 유형의 classification 샘플 데이터 생성. 
X_features, y_labels = make_classification(n_features=2, n_redundant=0, 
                                           n_informative=2,n_classes=3, 
                                           n_clusters_per_class=1,random_state=0)


# In[17]:


X_features


# In[18]:


y_labels


# In[19]:


X_features.shape


# In[20]:


# plot 형태로 2개의 feature로 2차원 좌표 시각화, 각 클래스값은 다른 색깔로 표시됨. 

plt.title('3 Class values with 2 Featires Sample data creation')
plt.scatter(X_features[:, 0], X_features[:, 1], marker='o', 
            c=y_labels, s=25, cmap='rainbow', edgecolor='k')

# 각 피처가 X, Y축으로 나열된 2차원 그래프
# 3개의 클래스 값 구분 : 3가지 색상


# ### 생성된 가상 데이터 세트 기반해서 시각화를 통한 데이터 분류 확인
# - 결정 트리 모델이 어떠한 결정 기준으로 분할하면서 데이터 분류하는지 확인
# - visualize_boundary() 함수 생성 : 클래스 값을 예측하는 결정 기준을 색상과 경계로 나타냄
#     - 모델이 어떻게 데이터 세트를 예측 분류하는지 보여줌
#         
# 1. 제약 없이 트리 생성. 하이퍼 파라미터 디폴트
#     - 이상치를 포함해서 분류되어, 기준 경계가 많아지고 복잡함   
#     - 복잡한 모델은 학습 데이터 특성과 조금만 달라져도 예측 정확도가 떨어짐
# 2. 하이퍼 파라미터 튜닝 : min_samples_leaf=6
#     - 노드 분류 규칙을 완화해서 복잡도 줄임
#     - 기준 경계 적어짐

# ![image.png](attachment:image.png)

# In[21]:


import numpy as np

# Classifier의 Decision Boundary를 시각화 하는 함수
# 클래스 값을 예측하는 결정 기준을 색상과 경계로 표시
def visualize_boundary(model, X, y):
    fig,ax = plt.subplots()
    
    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim()
    ylim_start , ylim_end = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행. 
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end,
                                                                               num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # contourf() 를 이용하여 class boundary 를 visualization 수행. 
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()), zorder=1)


# In[22]:


from sklearn.tree import DecisionTreeClassifier

# 특정한 트리 생성 제약없는 결정 트리의 Decsion Boundary 시각화.
dt_clf = DecisionTreeClassifier()
visualize_boundary(dt_clf, X_features, y_labels)


# In[23]:


# min_samples_leaf=6 으로 트리 생성 조건을 제약한 Decision Boundary 시각화
dt_clf = DecisionTreeClassifier(min_samples_leaf=6)
visualize_boundary(dt_clf, X_features, y_labels)


# ## 결정 트리 실습 - Human Activity Recognition

# 사용자 행동 인식 (Human Activity Recognition) 데이터 사용  
# 수집된 피처 세트를 기반으로 결정 트리를 이용해서 어떠한 동작인지 예측

# ### 사용자 행동 인식 (Human Activity Recognition) 데이터
# 
# - 19~48세 연령대 30명에게 스마트폰 센서를 장착
# - 허리에 스마트폰을(삼성 갤럭시 S II) 착용하고 6가지 활동을 수행
#     - WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
#     
# ![image-2.png](attachment:image-2.png)
# 
# - 내장 된 가속도계와 자이로스코프를 사용하여 50Hz의 일정한 속도로 3축 선형가속도와 3축 각속도를 캡처하여 동작과 관련된 속성 수집
# - 수동으로 레이블 지정을 위해 비디오로 녹화
# - 지원자의 70&가 훈련 데이터, 30%가 테스트 데이터
# - UCI Machine Learning Repository : https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
# 
# 
# ![image-4.png](attachment:image-4.png)
# 
# - features.txt :  561개 Feature 들어 있음 (index와 피처명)
#     - 피처명 : 인체의 움직임과 관련된 속성의 평균, 표준편차, ..,이 X, Y, Z축 값으로 되어 있음
#         - 가속도계의 3축 가속도 (총 가속도) 및 추정 된 신체 가속도
#         - 자이로 스코프로부터의 3 축 각속도
#         - 시간 및 주파수 도메인 변수가있는 561 특징 벡터
#         - 활동 라벨
#         - 실험을 수행 한 피험자의 식별자
#         
# - X_train.txt : 561개 각 feauture에 대해 맵핑되는 속성값이 들어 있음
# 

# ### features.txt 파일 로드 후 피처 정보 출력

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# features.txt 파일에는 피처 이름 index와 피처명이 공백(\s+)으로 분리되어 있어서
# sep='\s+' 옵션 필요, DataFrame으로 로드

feature_name_df = pd.read_csv('data/human_activity/features.txt', sep='\s+',
                             header=None, names=['column_index', 'column_name'])
feature_name_df


# In[26]:


# index 제외하고 피처명만 추출해서 리스트 객체로 생성한 뒤 샘플로 10개만 추출
feature_name = feature_name_df.iloc[:, 1].values.tolist()
feature_name[:10]


# ### 중복 피처 확인/변환

# **원본 데이터에 중복된 Feature 명으로 인하여 신규 버전의 Pandas에서 Duplicate name 에러를 발생.**  
# **중복 feature명에 대해서 원본 feature 명에 '_1(또는2)'를 추가로 부여하는 함수인 get_new_feature_name_df() 생성**

# In[3]:


# 중복된 피처명이 있어서, 
# 피처명을 변경해서 반환하는 함수 작성
# 전체 : 561개
# 중복 : 84개
# unique : 477개 


# In[27]:


# 중복되지 않은 피처명 (unique) 개수 확인
feature_name_df['column_name'].unique().shape


# In[29]:


# 중복된 피처명 count
# 그룹별로 count : 1보다 크면 중복

feature_dup_df = feature_name_df.groupby('column_name').count()
print(feature_dup_df[feature_dup_df['column_index']>1].count())

# 총 42개 피처명 중복
# 42 x 2 : 총 84개 중복
# 477 + 84 = 561

# 중복된 피처명과 그룹별 count 확인 
feature_dup_df[feature_dup_df['column_index']>1].head()

# 각 그룹에 3개씩


# **중복된 피처명 변경**

# In[30]:


####### 참고 : 중복된 피처명 변경하는 과정 ##########
# groupby('column_name').cumcount() : 중복되는 값이 몇 번째에 해당되는지(index) 반환
# 0이면 첫 번째, 1이면 두 번째, ...
# 중복된 피처명 순서값(index)을 컬럼으로 갖는 df 생성

f_dup_df = pd.DataFrame(data=feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
f_dup_df


# In[31]:


# merge하기 위해 index 값을 갖는 열 필요
# reset_index() : index를 열로 

f_dup_df_reset = f_dup_df.reset_index()
f_dup_df_reset


# In[32]:


feature_name_df


# In[33]:


new_feature_name_df = pd.merge(feature_name_df.reset_index(), f_dup_df_reset, how='outer' )
new_feature_name_df


# In[34]:


# 람다식 : 
# 입력 : ['column_name', 'dup_cnt'] 2개 칼럼 받아서
# 반환 : [1] 값(두 번째) dup_cnt'이 0보다 크면 [0] 값(첫 번째)_[1] 값(두 번째)
# 즉, column_name'_'dup_cnt'
# 아니면 [0] 값(첫 번째) 그대로 반환
new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) if x[1] >0 else x[0],  axis=1)
new_feature_name_df


# In[35]:


new_feature_name_df[new_feature_name_df['dup_cnt'] > 0]
# 피처명_1, 피처명_2로 변경된 것 확인


# In[12]:


new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
new_feature_name_df

########## 참고 끝


# ### 피처명 변경/반환하는 함수 get_new_feature_name_df()작성 

# In[37]:


# 피처명 변경해서 반환하는 과정을 함수로 작성
# 피처명_1 또는 피처명_2로 변경
# groupby('column_name').cumcount() : 중복되는 값이 몇 번째에 해당되는지(index) 반환
# 0이면 첫 번째, 1이면 두 번째, ...
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) 
                                                                                           if x[1] >0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df


# ### 사용자 행동 인식 데이터 세트 준비 함수 : get_human_dataset( )

# **아래 get_human_dataset() 함수는 중복된 feature명을 새롭게 수정하는   
# get_new_feature_name_df() 함수를 반영하여 수정**

# In[46]:


import pandas as pd

def get_human_dataset( ):
    
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당.
    feature_name_df = pd.read_csv('data/human_activity/features.txt',sep='\s+',
                        header=None,names=['column_index','column_name'])
    
    # 중복된 feature명을 새롭게 수정하는 get_new_feature_name_df()를 이용하여 새로운 feature명 DataFrame생성. 
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # DataFrame에 피처명을 컬럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    # 학습 피처 데이터 셋과 테스트 피처 데이터을 DataFrame으로 로딩. 컬럼명은 feature_name 적용
    X_train = pd.read_csv('data/human_activity/train/X_train.txt',sep='\s+', names=feature_name )
    X_test = pd.read_csv('data/human_activity/test/X_test.txt',sep='\s+', names=feature_name)
    
    # 학습 레이블과 테스트 레이블 데이터을 DataFrame으로 로딩하고 컬럼명은 action으로 부여
    y_train = pd.read_csv('data/human_activity/train/y_train.txt',sep='\s+',
                          header=None,names=['action'])
    y_test = pd.read_csv('data/human_activity/test/y_test.txt',sep='\s+',
                         header=None,names=['action'])
    
    # 로드된 학습/테스트용 DataFrame을 모두 반환 
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_human_dataset()


# **학습 피처 데이터셋 정보**

# In[39]:


print('## 학습 피처 데이터셋 info()')
print(X_train.info())
# 학습 데이터 세트 : 7352개 행, 561개 피처
# 피처가 전부 float형 숫자 : 카테고리 인코딩 작업 필요 없음


# In[40]:


# 학습용 피처 데이터 세트 확인
X_train.head()


# In[41]:


# null 값 확인
X_train.isna().sum().sum()
# null 값 없음


# In[42]:


# 학습용 레이블 데이터 세트 확인
y_train


# In[43]:


# 레이블 값 확인 
y_train['action'].unique()
# 레이블 값 : 1, 2, 3, 4, 5, 6


# In[44]:


# 레이블 값 분포 확인 (각 값의 개수)
print(y_train['action'].value_counts())
# 비교적 고르게 분포


# ### DecisionTreeClassifier를 이용해서 동작 예측 분류 수행

# In[48]:


# DecisionTreeClassifier를 이용해서 동작 예측 분류 수행
# 하이퍼 파라미터는 모두 디폴드 값으로 설정해서 수행
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 예제 반복 시 마다 동일한 예측 결과 도출을 위해 random_state 설정
dt_clf = DecisionTreeClassifier(random_state=156)

# 학습
dt_clf.fit(X_train, y_train)

# 예측
pred = dt_clf.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, pred)
print('결정트리의 예측정확도: ', accuracy)

# 결과
# 약 85.48 % 정확도


# In[49]:


# DecisionTreeClassifier의 하이퍼 파라미터 추출
dt_clf.get_params()


# ### 파라미터 튜닝 : GridSearchCV를 이용

# In[50]:


# 결정 트리의 깊이(Tree Depth)가 예측 정확도에 미치는 영향 확인
# 파라미터 튜닝 : GridSearchCV를 이용해서 max_depth 값을 계속 늘리면서 예측 성능 측정
# 교차 검증은 5개 세트 (cv=5)

from sklearn.model_selection import GridSearchCV

params = {
    'max_depth' : [6, 8 ,10, 12, 16 ,20, 24]
}

# 총 35회 학습 : 7 개의 max_depth 값 x 교차 검증 5개 세트 (cv)
# verbose : 장황함 : log 출력 level 조정 (출력 메시지 조절)
# verbose=1 : 간단한 메시지 출력
# verbose=0 : 메시지 출력 없이 결과만 출력 

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring='accuracy', cv=5, verbose=1 )

grid_cv.fit(X_train , y_train)
print('GridSearchCV 최고 평균 정확도 수치:{0:.4f}'.format(grid_cv.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터:', grid_cv.best_params_)

# 결과
# 최고 평균 정확도 : 85.13 % 도출
# 이때 최적 파라미터 값 : 16

# 이 예제의 수행 목표는 
# max_depth 값의 증가에 따라 예측 성능이 어떻게 변했는지 확인하는 것


# ### 수정 버전 01: 날짜 2019.10.27일  
# 
# **사이킷런 버전이 업그레이드 되면서 아래의 GridSearchCV 객체의 cv_results_에서 mean_train_score는 더이상 제공되지 않습니다.**  
# **기존 코드에서 오류가 발생하시면 아래와 같이 'mean_train_score'를 제거해 주십시요**
# 

# In[51]:


grid_cv.cv_results_


# In[53]:


cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df


# In[ ]:


# 5개의 CV 세트에서 max_depth 값에 따라 
# 예측이 성능이 어떻게 변했는지
# GridSearchCV객체의 cv_results_ 속성을 통해 확인
# cv_results_ : CV 세트에서 하이퍼 파라미터를 순차적으로 입력했을 때의 성능 수치
# cv_results_ 에서 추출할 값 : mean_test_score (평균 정확도 수치)

# GridSearchCV객체의 cv_results_ 속성을 DataFrame으로 생성. 
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
#cv_results_df


# In[55]:


# max_depth 파라미터 값과 그때의 테스트(Evaluation)셋, 학습 데이터 셋의 평균 정확도 수치 추출
cv_results_df[['param_max_depth', 'mean_test_score']]


# In[23]:


# 결과
# mean_test_score는 max_depth가 10일 때 0.851209로 평균 정확도가 정점이고
# 10을 넘으면 정확도가 계속 떨어짐

# 결정 트리는 도 완벽한 규칙을 학습 데이터 세트에 적용하기 위해
# 노드를 지속적으로 분할하면서 깊이가 깊어지고 더 복잡한 모델이 됨
# 깊어진 트리는 학습 데이터 세트에서는 올바른 예측 결과를 가져올지 모르지만
# 검증 데이터 세트에서는 오히려 과적합으로 인한 성능 저하 유발


# In[56]:


# 이번에는 별도의 테스트 데이터 세트에서 결정 트리의 정확도 측정
# max_depth 값을 변화 시키면서 그때마다 학습과 테스트 셋에서의 예측 성능 측정

max_depths = [ 6, 8 ,10, 12, 16 ,20, 24]

for depth in max_depths:
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=156)
    dt_clf.fit(X_train , y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test , pred)
    print('max_depth = {0} 정확도: {1:.4f}'.format(depth , accuracy))
    
# max_depth = 8 일 경우 정확도: 0.8707 이 가장 높음    
# 8을 넘어가면 정확도 계속 감속
# GridSearchCV 예제처럼 깊이가 깊어질수록 테스트 데이터 세트의 정확도 떨어짐

# 결정 트리의 깊이가 길어질수로 과적합 영향력이 커지므로
# 하이퍼 파라미터를 이용해서 깊이 제어 필요
# 복잡한 모델보다도 트리 깊이를 낮춘 단순한 모델이 더 효과적인 결과를 산출할 수 있음


# In[57]:


# max_depth와 min_samples_split을 같이 변경하면서
# 정확도 성능 튜닝

params = {
    'max_depth' : [ 8, 12, 16, 20], 
    'min_samples_split' : [16, 24]
}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring='accuracy', cv=5, verbose=1 )
grid_cv.fit(X_train , y_train)
print('GridSearchCV 최고 평균 정확도 수치: {0:.4f}'.format(grid_cv.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터:', grid_cv.best_params_)

# 결과
# max_depth:8,  min_samples_split: 16 일 때 가장 최고의 정확도 85.49 %


# In[61]:


# 별도 분리된 테스트 데이터 세트에 동일 하이퍼 파라미터 적용
# 앞 예제의 grid_cv 객체의 best_estimator_ 속성은
# 최적 하이퍼 파라미터 값 max_depth:8,  min_samples_split: 16 으로
# 학습이 완료된 Estimator 객체
# 이를 이용해서 테스트 데이터 세트에 예측 수행

best_dt_clf = grid_cv.best_estimator_

pred1 = best_dt_clf.predict(X_test)
print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred1)))

# 결과
# max_depth:8,  min_samples_split: 16일 때 
# 테스트 데이터 세트의 예측 정확도 : 약 87.17 %


# In[62]:


# 결정 트리에서 feature_importances_ 속성을 이용헤 각 피처의 중요도 확인
# 중요도가 높은 순으로 Top 20 피처를 막대그래프로 표현

import seaborn as sns

ftr_importances_values = best_dt_clf.feature_importances_

# Top 중요도로 정렬을 쉽게 하고, 시본(Seaborn)의 막대그래프로 쉽게 표현하기 위해 Series변환
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns  )

# 중요도값 순으로 Series를 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()


# In[29]:


ftr_importances_values # array


# In[30]:


ftr_importances # Series


# In[31]:


ftr_importances.index


# In[32]:


a =ftr_importances_values.argsort()
a


# In[33]:


import seaborn as sns

ftr_importances_values = best_df_clf.feature_importances_

# Top 중요도로 정렬을 쉽게 하고, 시본(Seaborn)의 막대그래프로 쉽게 표현하기 위해 Series변환
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns  )

# 중요도값 순으로 Series를 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()

