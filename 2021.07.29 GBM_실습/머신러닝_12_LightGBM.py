#!/usr/bin/env python
# coding: utf-8

# # LightGBM

# - XGBoost와 부스팅 계열 알고리즘에서 가장 각광을 받고 있음
# 
# - XGBoost 보다 학습에 걸리는 시간이 훨씬 적음
# 
# ### LightGBM의 장점
# - XGBoost 대비 더 빠른 학습과 예측 수행 시간
# - 더 작은 메모리 사용량
# - 카테고리형 피처의 자동 변환과 최적 분할
#     - 원-핫인코딩 등을 사용하지 않고도 카테고리형 피처를 최적으로 변환하고 이에 따른 노드 분할 수행
# 
# ### LightGBM의 단점
# - 적은 데이터 세트에 적용할 경우 과적합이 발생하기 쉬움
# - (공식 문서상 대략 10,000건 이하의 데이터 세트)
# 
# ### 기존 GBM과의 차이점
# - 일반적인 균형트리분할(Level Wise) 방식과 달리 **`리프중심 트리분할(Leaf Wise)`** 방식을 사용
# 
# - 균형트리분할은 최대한 균형 잡힌 트리를 유지하며 분할하여 트리의 깊이를 최소화하여 오버피팅에 강한구조이지만 균형을 맞추기 위한 시간이 필요함
# - 리프중심 트리분할의 경우 최대 손실 값을 가지는 리프노드를 지속적으로 분할하면서 트리가 깊어지고 비대칭적으로 생성
#     - 이로써 예측 오류 손실을 최소화하고자 함
# ![image.png](attachment:image.png)

# **사이킷런의 Estimator를 상속받아 fit(), predict() 기반의 학습과 예측, 사이킷런의 다양한 유틸리티 활용 가능**

# ## LightGBM 설치
# - 아나콘다를 통해 설치
# - 윈도우에 설치할 경우 Visual Studio tool 2015 이상이 먼저 설치되어 있어야 함
#     - https://visualstudio.microsoft.com/ko/downloads/
# ![image-2.png](attachment:image-2.png)
#     
# - 아나콘다 프롬프트를 관리자 권한으로 실행한 후 conda 명령어 수행
# 
# **`conda install -c conda-forge lightgbm`**
# ![image.png](attachment:image.png)

# In[1]:


import lightgbm

print(lightgbm.__version__)


# ### LightGBM 하이퍼파라미터

# - XGBoost와 매우 유사함
# - 유의할 점 :
#     - 리프 노드가 계속 분할하면서 트리의 깊이가 깊어지므로 이러한 트리 특성에 맞는 하이퍼 파라미터 설정이 필요
#     
# ![image.png](attachment:image.png)

# ### 하이퍼 파라미터 튜닝 방안
# 
# **[방안1] num_leaves 개수를 중심으로 min_child_samples(min_data_in_leaf), max_depth를 함께 조정하면서 모델의 복잡도를 줄이는 것**
# 
# - num_leaves는 개별 트리가 가질 수 있는 최대 리프 개수로 LightGBM 모델 복잡도를 제어하는 주요 파라미터
#     - num_leaves 개수를 높이면 정확도가 높아지지만, 반대로 트리의 깊이가 깊어지고 모델의 복잡도가 커져 과적합 영향도가 커짐
#         
#         
# - min_data_in_leaf는 사이킷런 래퍼 클래스에서는 min_child_samples로 변경됨
#     - 과적합을 개선하기 위한 중요한 파라미터
#     - num_leaves와 학습 데이터의 크기에 따라 달라지지만 보통 큰 값으로 설정하면 트리가 깊어짐을 방지함
#         
#         
# - max_depth는 명시적으로 깊이의 크기를 제한함
#     - num_leaves, min_data_in_leaf와 결합해 과적합을 개선하는데 사용
#     
#     
# **[방안2] learning_rate를 작게 하면서 n_estimations를 크게 하는 것**
# - n_estimators를 너무 크게 하는 것은 과적합으로 오히려 성능이 저하될 수 있으므로 주의
# 
# 
# 
# **[방안3] 과적합을 제어하기 위해서 reg_lambda, reg_alpha와 같은 regularization을 적용**
# 
# 
# **[방안4] 학습 데이터에 사용할 피처 수나 데이터 샘플링 레코드 개수를 줄이기 위해 colsample_bytree, subsample 파라미터를 적용하는 것도 과적합 제어 방안**
# 
#         
#         
# 

# ## 파이썬 래퍼 LightGBM, 사이킷런 래퍼 XGBoost, LightGBM 하이퍼파라미터 비교
# ![image.png](attachment:image.png)

# ## LightGBM 적용한 위스콘신 유방암 예측

# ### LightGBM의 파이썬 패키지인 lightgbm에서 LGBMClassifier 임포트

# In[2]:


# LightGBM의 파이썬 패키지인 lightgbm에서 LGBMClassifier 임포트
from lightgbm import LGBMClassifier


# ### 데이터 로드 및 학습/테스트 데이터 분할

# In[3]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
ftr = dataset.data
target = dataset.target

# 전체 데이터 중 80%는 학습용 데이터, 20%는 테스트용 데이터 추출
X_train, X_test, y_train, y_test=train_test_split(ftr, target, test_size=0.2, random_state=156 )


# ### LightGBM으로 학습

# In[5]:


# 앞서 XGBoost와 동일하게 n_estimators는 400 설정. 
lgbm_wrapper = LGBMClassifier(n_estimators=400)

evals =[(X_test, y_test)]

# 학습 : 조기 중단 수행 가능(XGBoost와 동일함)
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100,
                eval_metric='logloss', eval_set=evals, verbose=True)

# 예측
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]


#     -> 조기 중단으로 145번까지 반복을 수행하고 학습을 종료함

# ### LightGBM 기반 예측 성능 평가

# In[7]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))


# In[8]:


get_clf_eval(y_test, preds, pred_proba)


# ### plot_importance( )를 이용하여 feature 중요도 시각화

# In[9]:


# plot_importance( )를 이용하여 feature 중요도 시각화
from lightgbm import plot_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgbm_wrapper, ax=ax)


# In[ ]:




