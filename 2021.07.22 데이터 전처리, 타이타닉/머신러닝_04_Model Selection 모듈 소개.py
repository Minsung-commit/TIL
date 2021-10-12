#!/usr/bin/env python
# coding: utf-8

# # Model Selection 모듈 소개

# ## 학습/테스트 데이터 셋 분리 – train_test_split()

# ### 사이킷런 model_selection 모듈의 주요 기능
# - 학습 데이터와 테스트 데이터 세트 분리
# - 교차 검증 분할 및 평가
# - Estimator의 하이퍼 파라미터 튜닝

# **학습 데이터와 테스트 데이터 세트 분리**
# - train_test_split() 함수 사용
# 
# **학습 데이터 세트**
# - 머신러닝 알고리즘의 학습을 위해 사용
# - 데이터의 속성(피처)과 결정값(레이블) 모두 포함
# - 학습 데이터를 기반으로 머신러닝 알고리즘이 데이터 속성과 결정값의 패턴을 인지하고 학습
# 
# **테스트 데이터 세트** 
# - 학습된 머신러닝 알고리즘 테스트용
# - 머신러닝 알고리즘은 제공된 속성 데이터를 기반으로 결정값 예측
# - 학습 데이터와 별도의 세트로 제공

# **train_test_split() 함수**
# 
# train_test_split(feature_dataset, label_dataset, test_size, train_size, random_state, shuffle, stratify)
# 
# 
# - feature_dataset : 피처 데이터 세트
#     - 피처(feature)만으로 된 데이터(numpy) [5.1, 3.5, 1.4, 0.2],...
# - label_dataset : 레이블 데이터 세트
#     - 레이블(결정 값) 데이터(numpy) [0 0 0 ... 1 1 1 .... 2 2 2]
# - label_dataset : 테스트 데이터 세트 비율
#     - 전체 데이터 세트 중 테스트 데이터 세트 비율
#     - 지정하지 않으면 0.25
# - random_state : 세트를 섞을 때 해당 int 값을 보고 섞음
#     - 수행할 때마다 동일한 데이터 세트로 분리하기 위해 시드값 고정(실습용)
#     - 0 또는 4가 가장 많이 사용
#     - 하이퍼 파라미터 튜닝시 이 값을 고정해두고 튜닝해야 매번 데이터셋이 변경되는 것을 방지할 수 있음
#     
# - shuffle : 분할하기 전에 섞을지 지정
#     - default=True (보통은 default 값으로 놔둠)
# - stratify : 지정된 레이블의 클래스 비율에 맞게 분할
#     - default=None
#     - classification을 다룰 때 매우 중요한 옵션값
#     - stratify 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train/ validation에 유지해 줌(한 쪽에 쏠려서 분배되는 것을 방지)
#     - 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있음
#     
#    
#    
# 예. train_test_split(iris_data, iris_label, test_size=0.3, random_state=11)
# 

# train_test_split() 반환값
# * X_train : 학습용 피처 데이터 세트 (feature)
# * X_test : 테스트용 피처 데이터 세트 (feature)
# * y_train : 학습용 레이블 데이터 세트 (target)
# * y_test : 테스트용 레이블 데이터 세트 (target)
# * feature : 대문자 X_
# * label(target) : 소문자 y_

# ### (1) 학습/테스트 데이터 셋 분리하지 않고 예측

# In[1]:


# (1) 학습/테스트 데이터 셋 분리하지 않고 예측

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris_data = load_iris()
dt_clf = DecisionTreeClassifier()

train_data = iris_data.data
train_label = iris_data.target

# 학습 수행
dt_clf.fit(train_data, train_label)

# 테스트
pred = dt_clf.predict(train_data)
print("예측 정확도:", accuracy_score(train_label, pred))


# - 예측을 train_data로 했기 때문에 결과 1.0 (100%)으로 출력 (잘못됨)
# - 예측은 테스트 데이터로 해야 함

# ### (2) 학습/테스트 데이터 셋 분리하고 예측

# In[13]:


# (2) 학습/테스트 데이터 셋 분리하고 예측 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris_data = load_iris()
dt_clf = DecisionTreeClassifier()

# 학습/테스트 분할(split)
X_train, X_test, y_train, y_test = train_test_split(iris_data.data,
                                                    iris_data.target,
                                                    test_size=0.2,
                                                    random_state=4)
print(y_train)


# In[14]:


# 학습 수행
dt_clf.fit(X_train, y_train)

# 예측 수행
pred = dt_clf.predict(X_test)
print("예측정확도:", accuracy_score(y_test, pred))


# 넘파이 ndarray 뿐만 아니라 판다스 DataFrame/Series도 train_test_split( )으로 분할 가능

# In[15]:


import pandas as pd

iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['target']=iris_data.target
iris_df.head(3)


# In[18]:


# 피처 데이터프레임 반환 (마지막 열 전까지, 마지막 열 제외)
feature_df = iris_df.iloc[:, :-1]

# 타깃 데이터프레임 반환
target_df = iris_df.iloc[:, -1]

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(feature_df,
                                                    target_df,
                                                    test_size=0.3,
                                                    random_state=4)


# In[19]:


type(X_train)


# In[23]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print('예측정확도: {0:.3f}'.format(accuracy_score(y_test, pred)))


# ## Data Split과 모델 검증
# 
# - 언제
#     - "충분히 큰" 데이터 세트를 가용할 때
#     - "충분히 큰" 데이터가 없을 때에는 교차 확인(Cross Validation) 고려
#     
# 
# - 왜
#     - 학습에 사용되지 않은 데이터를 사용하여 예측을 수행함으로써 모델의 일반적인 성능에 대한 적절한 예측을 함
#     
# 
# - 어떻게
#     - 홀드-아웃(Hold-out)
#     - 교차검증(Cross Validation,CV)
#     - 필요에 따라 Stratified Sampling

# ### 홀드-아웃 방식
# - 데이터를 두 개 세트로 나누어 각각 Train과 Test 세트로 사용
# - Train과 Test의 비율을 7:3 ~ 9:1로 널리 사용하나, 알고리즘의 특성 및 상황에 따라 적절한 비율을 사용
# - Train – Validation - Test로 나누기도 함
# 
# ![image.png](attachment:image.png)
# https://algotrading101.com/learn/train-test-split-2/
# 

# 부적합한 데이터 선별로 인한 문제점
# - ML은 데이터에 기반하고, 
# - 데이터는 이상치, 분포도, 다양한 속성값, 피처 중요도 등 
# - ML에 영향을 미치는 다양한 요소를 가지고 있음
# - 특정 ML 알고리즘에 최적으로 동작할 수 있도록
# - 데이터를 선별해서 학습한다면
# - 실제 데이터 양식과는 많은 차이가 있을 것이고
# - 결국 성능 저하로 이어질 것임
# 
# 문제점 개선 ---> 교차 검증을 이용해 더 다양한 학습 평가 수행

# ### 교차검증(Cross Validation, CV)
# - k-fold Cross Validation이라고도 함
# - 전체 데이터 세트를 임의로 k개의 그룹으로 나누고, 그 가운데 하나의 그룹을 돌아가면서 테스트 데이터 세트로, 나머지 k-1개 그룹은 학습용 데이터 세트로 사용하는 방법
# - 별도의 여러 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 학습과 평가를 수행
# 
# 
# - 사용 목적
#     - 데이터에 적합한 알고리즘인지 평가하기 위해 
#     - 모델에 적절한 hyperparameter 찾기 위해
#     - 과대적합 예방
#     - 데이터 편중을 막기 위해
#     
# 
#     
# ![image.png](attachment:image.png)
# 
# 
# http://karlrosaen.com/ml/learning-log/2016-06-20/

# ### 교차 검증 방법
# - K 폴드 교차 검증
# - Stratified K 폴드 교차 검증

# ### K 폴드 교차 검증
# - K개의 데이터 폴드 세트를 만들어서
# - K번만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행
# - 가장 보편적으로 사용되는 교차 검증 기법
# 
# 
# - 5-폴드 교차 검증
# 
# ![image.png](attachment:image.png)

# **K 폴드 교차 검증 프로세스 구현을 위한 사이킷런 클래스**
# 
# (1) KFold 클래스 : 폴드 세트로 분리하는 객체 생성
# - kfold = KFold(n_splits=5)
# 
# (2) split() 메소드 : 폴드 데이터 세트로 분리
# - kfold.split(features)
# - 각 폴드마다  
#     학습용, 검증용, 테스트 데이터 추출  
#     학습용 및 예측 수행  
#     정확도 측정  
#     
# (3) 최종 평균 정확도 계산

# * K 폴드 예제

# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

iris = load_iris()
features = iris.data
label = iris.target

features.shape


# In[34]:


# DecisionTreeClassifier 객체 생성 
dr_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체 생성
kfold = KFold(n_splits=5)

# 폴드 세트별 정확도를 담을 리스트 객체 생성
cv_accuracy = []


# In[35]:


# 폴드 별 학습용, 검증용 데이터 세트의 행 인덱스 확인
for train_index, test_index in kfold.split(features):
    print(train_index, test_index)


# In[36]:


import numpy as np

for train_index, test_index in kfold.split(features):
    X_train = features[train_index]
    X_test = features[test_index]
    y_train = label[train_index]
    y_test = label[test_index]
    
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    acc = np.round(accuracy_score(y_test, pred), 3)
    
    train_size= X_train.shape[0]
    test_size = X_test.shape[0]
    
    print('정확도: %f, 학습데이터크기: %d, 검증데이터크기: %d' %(acc, train_size, test_size))
          
    cv_accuracy.append(acc)
    
print('평균 검증 정확도: ', np. mean(cv_accuracy))


# ### Stratified K 폴드 교차 검증
# - 불균형한 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K 폴드 방식

# ### 불균형한 데이터(imbalanced data) 문제
# - 관심 대상 데이터가 상대적으로 매우 적은 비율로 나타나는 데이터 문제
# 
# - 분류 문제인 경우 : 클래스들이 균일하게 분포하지 않은 문제를 의미
#     - 예. 불량률이 1%인 생산라인에서 양품과 불량품을 예측하는 문제
#     - 사기감지탐지(fraud detection), 이상거래감지(anomaly detection), 의료진단(medical diagnosis) 등 에서 자주 나타남
# 
# - 회귀 문제인 경우 : 극단값이 포함되어 있는 "치우친" 데이터 사례
#     - 예. 산불에 의한 피해 면적을 예측하는 문제
#     (https://www.kaggle.com/aleksandradeis/regression-addressing-extreme-rare-cases)
# 
# 
# **우회/극복하는 방법**
# - 데이터 추가 확보
# 
# 
# - Re-Sampling
#     - Under-sampling(과소표집)
#         - 다른 클래스에 비하여 상대적으로 많이 나타나는 클래스의 개수를 줄임
#         - 균형은 유지할 수 있으나 유용한 정보에 대한 손실이 있을 수 있음
#     - Over-Sampling(과대표집)
#         - 상대적으로 적게 나타나는 클래스의 데이터를 복제하여 데이터의 개수를 늘림
#         - 정보 손실은 없이 학습 성능은 높아지는 반면, 과적합의 위험이 있음
#         - 이를 회피하기 위해서 SMOTE 와 같이 임의의 값을 생성하여 추가하는 방법 사용
#         
#         
# ![image.png](attachment:image.png)    

# * 먼저 K 폴드 문제점 확인하고, 
# * 사이킷런의 Stratified K 폴드 교차 검증 방법으로 개선
# * 붓꽃 데이터 세트를 DataFrame으로 생성하고 
# * 레이블 값의 분포도 확인

# In[37]:


import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df.head()


# In[38]:


iris_df['label'].value_counts()
# 레이블 값은 0, 1, 2 값 모두 50개로 동일
# 즉, Setosa, Versicolor, virginica 각 품종 50개 씩 


# In[43]:


# 3개 폴드를 구성
kfold = KFold(n_splits=3)

n=0
for train_index, test_index in kfold.split(iris_df):
    n += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print("[교차검증: %d]" %(n))
    print("  학습용 : \n", label_train.value_counts())
    print("  검증용 : \n", label_test.value_counts())


# In[ ]:


########## 참고 : 3개의 폴드 세트로 KFold 교차 검증 : 정확도 : 0 ###########


# In[46]:


# DecisionTreeClassifier 객체 생성 
dt_clf = DecisionTreeClassifier(random_state=156)

# 3개의 폴드 세트로 분리하는 KFold 객체 생성
kfold = KFold(n_splits=3)

# 폴드 세트별 정확도를 담을 리스트 객체 생성
cv_accuracy = []

n = 0

for train_index, test_index in kfold.split(iris_df):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
    n += 1
    
    # 반복 시 마다 정확도 측정 
    acc = np.round(accuracy_score(y_test,pred), 3)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('%d \n정확도: %f, 학습데이터크기: %d, 검증데이터크기: %d' %(n, acc, train_size, test_size))
          
    cv_accuracy.append(accuracy)    
    
# 개별 iteration별 정확도를 합하여 평균 정확도 계산 
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy)) 


# In[ ]:


########## 참고 끝   ###########


# - 위 코드 결과의 문제점
#     - 학습하지 않은 데이터를 검증 데이터로 사용
#     - 원할한 학습과 예측이 어려움
#     - 검증 정확도는 0

# StratifiedKFold 클래스
# - 원본 데이터의 레이블 분포를 고려한 뒤 이 분포와 동일하게 학습과 검증데이터 세트를 분배
# 
# - KFold 사용법과 거의 비슷
# - 차이점
#   - 레이블 데이터 분포도에 따라 학습/검증 데이터를 나누기 때문에
#   - split() 메서드에 인자로 피처 데이터 세트뿐 아니라 
#   - 레이블 데이터 세트도 반드시 필요하다는 것

# In[47]:


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n=0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n = n + 1
    
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print("[교차검증 : %d] % n")
    print('학습용 레이블 분포: \n', label_train.value_counts())
    print('검증용 레이블 분포: \n', label_test.value_counts())


# In[49]:


# StratifiedKFold를 이용해 붓꽃 데이터 교차 검증

dt_clf = DecisionTreeClassifier(random_state=156)

# 3개의 폴드 세트로 분리하는 StratifiedKFold 객체 생성
skfold = StratifiedKFold(n_splits=3)

# 폴드 세트별 정확도를 담을 리스트 객체 생성
cv_accuracy = []

n = 0

for train_index, test_index in skfold.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
        
    # 반복 시 마다 정확도 측정 
    n += 1
       
    acc = np.round(accuracy_score(y_test,pred), 3)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('%d \n정확도: %f, 학습데이터크기: %d, 검증데이터크기: %d' %(n, acc, train_size, test_size))
          
    cv_accuracy.append(acc)    
    
# 개별 iteration별 정확도를 합하여 평균 정확도 계산 
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy)) 


# Stratified K 폴드의 경우
# - 원본 데이터의 레이블 분포도 특성을 반영한 학습 및 검증 데이터 세트를 만들 수 있으므로
# - 왜곡된 레이블 데이터 세트에서는 반드시 Stratified K 폴드를 이용해서 교차 검증해야 함
# - 일반적으로 분류(Classification)에서의 교차 검증은 K 폴드가 아니라 Stratified K 폴드로 분할되어야 함
# - 회귀(Regression)에서는 Stratified K 폴드 지원되지 않음
#     - 이유 : 회귀의 결정값은 이산값 형태의 레이블이 아니라 연속된 숫자값이기 때문에
#     - 결정값별로 분포를 정하는 의미가 없기 때문

# ## 교차검증을 보다 간편하게 

# - 교차 검증 (Cross Validation) 과정
#     1. 폴드 세트 설정
#     2. for 문에서 반복적으로 학습 및 검증 데이터 추출 및 학습과 예측 수행
#     3. 폴드 세트별로 예측 성능을 평균하여 최종 성능 평가

# ### cross_val_score( ) 함수
# - 1 ~ 3 단계의 교차 검증 과정을 한꺼번에 수행
# - 내부에서 Estimator를 학습(fit), 예측(predict), 평가(evaluation) 시켜주므로
# - 간단하게 교차 검증 수행 가능

# ![image.png](attachment:image.png)

# ### 붓꽃 자료를 3개 폴드로 분할하여 학습 및 검증

# In[50]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score , cross_validate
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

features = iris.data
label = iris.target

scores = cross_val_score(dt_clf, features, label, scoring='accuracy', cv=3)
print('교차 검증별 정확도:', scores)
print('평균 검증 정확도:', np.round(np.mean(scores), 4))


# - cross_val_score()는 cv로 지정된 횟수만큼
# - scoring 파라미터로 지정된 평가 지표로 평가 결과값을 배열로 반환
# - 일반적으로 평가 결과값 평균을 평가 수치로 사용

# ## 교차 검증과 최적의 하이퍼파라미터 튜닝을 한번에

# 하이퍼파라미터(Hyper parameter)
# - 머신러닝 알고리즘을 구성하는 요소
# - 이 값들을 조정해 알고리즘의 예측 성능을 개선할 수 있음

# ### 사이킷런의 GridSearchCV클래스

# - Classifier나 Regressor와 같은 알고리즘에 사용되는
# - 하이퍼 파라미터를 순차적으로 입력하면서
# - 최적의 파라미터를 편리하게 도출할 수 있는 방법 제공  
# -(Grid는 격자라는 의미 : 촘촘하게 파라미터를 입력하면서 테스트 하는 방식)
# 
# 즉,  
# - 머신러닝 알고리즘의 여러 하이퍼 파라미터를  
# - 순차적으로 변경하면서 최고 성능을 가지는 파라미터를 찾고자 한다면  
# - 파라미터의 집합을 만들어 순차적으로 적용하면서 최적화 수행  

# **GridSearchCV 클래스 생성자의 주요 파라미터**
# 
# - estimator : classifier, regressor, peipeline
# 
# 
# - param_grid : key + 리스트 값을 가지는 딕셔너리 (estimator 튜닝을 위한 하이퍼 파라미터 )
#      - key: 파라미터명, 리스트값:파라미터 값
#      
#      
# - scoring : 예측 성능을 측정할 평가 방법 
#      - 성능 평가 지표를 지정하는 문자열
#      - 예: 정확도인 경우 'accuracy'
#      
#      
# - cv : 교차 검증을 위해 분할되는 학습/테스트 세트의 개수
# 
# 
# - refit : 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습 여부
#      - 디폴트 : True    
# 

# In[57]:


# GridSearchCV를 이용해
# 결정 트리 알고리즘의 여러 가지 최적화 파라미터를 순차적으로 적용해서
# 붓꽃 데이터 예측 분석

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                   iris.target,
                                                   test_size=0.2,
                                                    random_state=121)

dt_clf = DecisionTreeClassifier()

parameters = {'max_depth': [1, 2, 3], 'min_samples_split':[2, 3]}
# 하이퍼파라미터는 딕셔너리 형식으로 지정
# key : 결정트리의 하이파라미터
# value : 하이퍼파라미터의 값


# min_samples_split : 자식 규칙 노드를 분할해서 만드는데 필요한 최소 샘플 데이터 개수
# - min_samples_split=4로 설정하는 경우
#     - 최소 샘플 개수가 4개 필요한데
#     - 3개만 있는 경우에는 더 이상 자식 규칙 노드를 위한 분할을 하지 않음
# 
# 
# 트리 깊이도 줄어서 더 간결한 결정 트리 생성
# 
# 
# ![image.png](attachment:image.png)

# In[59]:


# 
grid_tree = GridSearchCV(dt_clf, param_grid=parameters, cv=3, refit=True, return_train_score=True)

grid_tree.fit(X_train, y_train)

scores_df = pd.DataFrame(grid_tree.cv_results_)
scores_df


# In[60]:


# 파라미터 확인
grid_tree.cv_results_


# In[61]:


# GridSearchCV 결과 세트로 딕셔너리 형태인 cv_results_ 를 
# DataFrame으로 변환 후 # 일부 파라미터 확인

scores_df[['params', 'mean_test_score', 'rank_test_score']]


# In[63]:


# 최고 성능을 가지는 파라미터 조합 및 예측 성능 1위 값 출력
print('최적 파라미터: ', grid_tree.best_params_)
print('최고 정확도:', grid_tree.best_score_)


# In[65]:


# GridSearchCV 객체의 생성 파라미터로 refit=True로 설정된 경우(디폴트)
# GridSearchCV가 최적 성능을 나타내는 하이퍼 파라미터로 Estimator를 학습하고
# best_estimator_ 로 저장
# (GridSearchCV의 refit으로 이미 학습이 된 estimator) 
best_dt = grid_tree.best_estimator_

# (best_estimator_는 이미 최적 학습이 됐으므로 
# 별도 학습 필요 없이 바로 예측 가능)

pred = best_dt.predict(X_test)
accuracy_score(y_test, pred)


# **일반적인 머신러닝 모델 적용 방법**
# 
# - 일반적으로 학습 데이터를 GridSearchCV를 이용해
# - 최적 하이퍼 파라미터 튜닝을 수행한 뒤에
# - 별도의 테스트 세트에서 이를 평가하는 방식
