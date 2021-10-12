#!/usr/bin/env python
# coding: utf-8

# # 회귀(Regression)

# ## 회귀분석 소개

# : 분류와 더불어 지도학습의 가장 큰 축 중의 하나
#    
# - 현대 통계학을 이루는 주요 기둥 중 하나
# 
# - 경제학, 사화과학, 의학, 공학 등 대부분의 분야에서 회귀가 많이 활용되고 있음
# 
# 
# - 인과 관계를 파악하는 분석
#     - 예. 마케팅 비용을 늘리면 매출이 늘어나는가?
#     
# 
# - 분류보다 회귀가 현업 업무 분석가들이나 데이터 분석가들이 많이 활용
#     - 대부분 기업에서 사용하는 예측이 숫자적 예측이 많기 때문
#     
# 
# - 우리는 머신러닝 관점에서의 회귀 사용
#     - 예측 : 설명변수로부터 결과변수를 예측하는 모델(함수식)을 구함

# ## 회귀(regrssion)
# 
# - 사전적 의미 “go back to an earlier and worse condition” (옛날 상태로 돌아감)
# - 평균으로의 회귀(regression toward mean)
# - 영국의 유전학자 Francis Galton(1822~1911)의 연구에서 유래
#     - Sweet pea    
# 
#     - 부모 키와 자녀 키의 관계 연구
#         - 928명의 성인 자녀 키(여자는 키에 1.08배, 행)와 
#         - 부모 키(아버지 키와 어머니 키의 평균, 열)을 조사하여 다음 표를 얻음
#         - Galton은 표를 관찰한 결과 :
#             - 키는 일정한 수준이상이면 무한정 커지는 것이 아니며
#             - 일정 수준 이하이면 무한정 작아지는 것이 아니라 
#             - 전체 키 평균으로 돌아가려는 경향이 있음을 확인
#             - 표에서 직선이 중앙 일정 구간 이외에는 중심으로 향해 낮아짐
# 
# 
# ![image-6.png](attachment:image-6.png) 
# http://galton.org/statistician.html            
#             
# - **중심으로 회귀**하려는 경향
# - **회귀분석**이라 명명

# - Karl Pearson(1903)
#     - Galton의 아이디어를 수리적 모형화 및 추정 방법 제안
#     - OLS(Ordinary Least Square, 최소제곱법)
#     - 1078명 아버지와 아들의 키 데이터 활용
#         - Son_H = 33.73 + 0.516 * F_H

# ### 분류와 회귀 차이점 : 결과값의 차이
# 
# 분류 : category 반환
# - 0/1, 이미지1/이미지2, 카테고리값 0, 1, 2, negative/positive
# - 사자, 고양이, 강아지
# 
# 회귀 : 숫자값
# - 아파트 값 예측
# - 매출 증가액 예측

# ### 회귀모델 유형
# - 선형/비선형: 회귀계수 결합에 따라
# - 단순/다중  : 독립변수 개수에 따라
# 
# **OLS 회귀분석 종류**
# - 단순회귀분석(simple regression analysis)
#     - 하나의 독립변수로 하나의 종속변수를 설명하는 모형
#     - 예. 아버지의 키로 한 자녀의 키를 설명하는 경우
# - 다중회귀분석(multiple regression analysis)
#     - 두 개 이상의 독립변수로 하나의 종속변수를 설명하는 모형 
#     - 예. 아버지와 어머니의 키로 한 자녀의 키를 설명
# - 다항회귀분석(polynimial regression analysis)
#     - 독립변수와 종속변수의 관계를 2차 이상의 함수로 설명
#     - 예. 2차 함수관계 →  독립변수=$(x, x^2 )$ → 독립변수 간의 종속성에 주의
# - 다변량회귀분석(multivariate regression analysis)
#     - 두 개 이상의 종속변수를 사용하는 모형 
#     - 예. 아버지와 어머니의 키로 두 자녀의 키를 설명하는 경우
#     
#     
# ![image-2.png](attachment:image-2.png)

# ## 선형 회귀(Linear Regression)

# ### 목적
# 
# - 설명 : 종속변수에 대한 설명(독립)변수의 영향을 측정, 설명함
# - 예측 : 모델 함수식을 통해 설명(독립)변수 정보에 따른 종속변수의 값을 예측함
# 
# 
# ![image-3.png](attachment:image-3.png)

# **여러 개의 독립변수와 한 개의 종속변수 간의 상관관계를 모델링하는 기법**
# 
# - $ Y_i = β_0 + β_1 X_{1i} + β_2 X_{2i} + β_3 X_{3i} + ... + β_n X_{ni} $
# 
# 
# - 독립 변수 ($X_i $) : 서로 간 상관관계가 적고 독립립적인 변수.
#     - 피처.(아파트 크기, 방 개수, 주변 지하철역 수, 주변 학군)
# 
# 
# - 종속 변수  ($Y$): 독립 변수의 영향을 받는 변수. 
#     - 결정값. (아파트값)
# 
# 
# - 회귀 계수 ($β_i $) : 변수의 값에 영향을 미치는 것  

# **실제 값과 예측값의 차이를 최소화하는 직선형 회귀선을 최적화하는 방식**
# 
# : 오차의 제곱합
# 
# - $ e_i = y_i - \hat {y_i} $
# 
# 
# - $ min \sum { e_i^2 } $

# ### 선형회귀분석의 주요 가정
# 
# - 선형성
# - 정규성
# - 등분산성
# - 독립성
# 
#  $ Y_i = β_0 + β_1 X_{1i} + β_2 X_{2i} + β_3 X_{3i} + ... + β_n X_{ni} + ε_i $
#  
#  $ ε_i -  N(0, σ^2 ) $
#  
#  ![image.png](attachment:image.png)
#  
#  https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/R/R5_Correlation-Regression/R5_Correlation-Regression4.html

# ### 대표적인 선형 회귀 모델
# - 일반 선형 회귀 : 규제를 적용하지 않은 모델로 예측값과 실졔값의 RSS를 최소화할 수 있도록 최적화
# - 릿지 (Ridge) : 선형 회귀에  L2규제를 추가한 회귀 모델
# - 라쏘 (Lasso) : 선형 회귀에  L1규제를 적용한 모델
# - 엘라스틱넷 (ElasticNet) : L2, L1 규제를 함께 결합한 모델
# - 로지스틱 회귀 (Logistic Regression) : 분류에서 사용되는 선형 모델

# *규제 : 일반적인 선형 회귀의 과적합 문제를 해결하기 위해서 회계 계수에 패널티 값을 적용하는 것

# ## 단순 선형 회귀를 통한 회귀 이해

# - 독립변수(피처)가 하나뿐인 선형회귀
# 
#     - $ y_i = w_0 + w_1 X_i + Error_i $
#     - 위 식은 β대신에 w를 사용 (머신러닝에서는 회귀계수를 가중치(weight)로 표현) 
#     
# 
# 
# **예. 주택 가격이 단순히 주택의 크기로만 결정되는 단순선형회귀로 가정**
# 
# 
# - 주택 가격은 주택 크기에 대해 선형(직선 형태)의 관계로 표현
# 
# 
# - 그래프로 표시 
#     - X 축 : 주택 크기 (평수)
#     - Y 축 : 주택 가격        
#         
# ![image-2.png](attachment:image-2.png)
# 
# 
# 
# **최적의 회귀 모델**
# - 실제값과 모델 사이의 오류값(잔차)이 최소가 되는 모델
# - 오류 값 합이 최소가 될 수 있는 최적의 회귀 계수를 찾는 것
# 
# -  $ RSS(w_0, w_1 ) = {\frac{1}{N}}  \sum_{i=1}^N (y_i - (w_0 + w_1 * x_i ))^2 $
# 
# 
# => 머신러닝에서는 **`비용함수(Cost Function)`**라고 하며, **`손실함수(loss function)`**이라고도 함

# # 비용 최소화하기 : 경사하강법(Gradient Descent)

# 비용함수(손실함수)가 최소가 되는 지점의 w를 계산
# - 2차 함수의 미분값인 1차 함수의 기울기가 가장 최소일 때
# - 최초 w에서부터 미분을 적용한 뒤 이 미분값이 계속 감소하는 방향으로 순차적으로 업데이트
# - 더 이상 미분된 1차 함수의 기울기가 감소하지 않는 지점으로 비용함수의 최소인 지점으로 간주하고 w를 반환
# - 비용함수의 계수가 두 개이상이므로 편미분
# 
# 
# ![image-2.png](attachment:image-2.png)

# [참고] 경사하강법(Gradient descent)
# 
# ![image-4.png](attachment:image-4.png)
# 
# https://machinelearningmedium.com/2017/08/15/gradient-descent/

# ### 경사하강법의 일반적인 프로세스
# 
# - [Step1]  $ w_0 , w_1 $를 임의의 값으로 설정하고 첫 비용함수의 값을 계산한다
# 
# 
# - [Step2]  $ w_1 $를 $ w_1 + η \frac {2}{N} \sum_{i=1}^N x_i *(실제값_i - 예측값_i ) $ 으로 $ w_0 $를 $ w_0 + η \frac {2}{N} \sum_{i=1}^N (실제값_i - 예측값_i ) $ 으로 업데이트 한 후 다시 비용함수의 값을 계산한다.
# 
# 
# - [Step3]  비용함수의 값이 감소했으면 다시 Step2를 반복한다. 더 이상 비용 함수의 값이 감소하지 않으면 그 때의 $ w_1, w_0 $를 구하고 반복을 중지한다.

# **실제값을 Y=4X+6 시뮬레이션하는 데이터 값 생성**

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)

# y = 4X + 6 식을 근사(w1=4, w0=6). random 값은 Noise를 위해 만듬
X = 2 * np.random.rand(100,1)
y = 6 + 4 * X + np.random.randn(100,1)

# X, y 데이터 셋 scatter plot으로 시각화
plt.scatter(X, y)


# In[2]:


X.shape, y.shape


# **$w_0$과 $w_1$의 값을 최소화 할 수 있도록 업데이트 수행하는 함수 생성**
# 
# * 예측 배열 y_pred는 np.dot(X, w1.T) + w0 임
#     - 100개의 데이터 X(1,2,...,100)이 있다면 
#     - 예측값은 $w_0 + X(1)*w_1 + X(2)*w_1 +..+ X(100)*w_1$이며, 
#     - 이는 입력 배열 X와 $w_1$ 배열의 내적임.
#     
# * 새로운 w1과 w0를 update함

# In[1]:


# w1 과 w0 를 업데이트 할 w1_update, w0_update를 반환. 

def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y)
    
    # 먼저 w1_update, w0_update를 각각 w1, w0의 shape와 동일한 크기를 가진 0 값으로 초기화
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    
    # 예측 배열 계산하고 예측과 실제 값의 차이 계산
    y_pred = np.dot(X, w1.T) + w0
    diff = y-y_pred
         
    # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성 
    w0_factors = np.ones((N,1))

    # w1과 w0을 업데이트할 w1_update와 w0_update 계산
    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))    
    
    return w1_update, w0_update


# In[3]:


w0 = np.zeros((1,1))
w1 = np.zeros((1,1))

y_pred = np.dot(X, w1.T) + w0
diff = y-y_pred
print(y_pred, diff)

w0_factors = np.ones((100,1))
w1_update = -(2/100)*0.01*(np.dot(X.T, diff))
w0_update = -(2/100)*0.01*(np.dot(w0_factors.T, diff))   

print(w1_update, w0_update)
print(w1_update.shape, w0_update.shape)


# **반복적으로 경사 하강법을 이용하여 get_weight_updates()를 호출하여 $w_1$과 $w_0$를 업데이트 하는 함수  `gradient_descent_steps()` 생성**

# In[4]:


# 입력 인자 iters로 주어진 횟수만큼 반복적으로 w1과 w0를 업데이트 적용함. 

def gradient_descent_steps(X, y, iters=10000):
    # w0와 w1을 모두 0으로 초기화. 
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    
    # 인자로 주어진 iters 만큼 반복적으로 get_weight_updates() 호출하여 w1, w0 업데이트 수행. 
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
              
    return w1, w0


# **예측 오차 비용을 계산을 수행하는 함수 `get_cost()` 생성 및 경사 하강법 수행**

# In[5]:


def get_cost(y, y_pred):
    N = len(y) 
    cost = np.sum(np.square(y - y_pred))/N
    return cost


# In[6]:


w1, w0 = gradient_descent_steps(X, y, iters=1000)
print("w1:{0:.3f} w0:{1:.3f}".format(w1[0,0], w0[0,0]))

y_pred = w1[0,0] * X + w0
print('Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))


# **경사하강법으로 구한 선형식을 시각화**

# In[7]:


plt.scatter(X, y)
plt.plot(X, y_pred)


# ### 미니배치 확률적 경사 하강법(Stochastic Gradient Descent)을 이용한 최적 비용함수 도출

# - 일반적인 경사하강법은 모든 학습 데이터에 대해 반복적으로 비용함수 최소화를 위한 값을 업데이트하기 때문에 수행 시간이 매우 오래걸리는 단점이 있음
# 
# - 확률적 경사하강법(Stochstic Gradient Descent)
#     - 전체 입력 데이터로 w가 업데이트되는 값을 계산하는 것이 아니라 일부 데이터만 이용해 w가 업데이트되는 값을 계산함
#     - 경사하강법에 비해 빠른 속도를 보장
#     - 대용량의 데이터의 경우 확률적 경사하강법이나 미니배치 확률적 경사하강법을 이용해 최적 비용함수를 도출함

# **미니 배치 확률적 경사하강법 함수 `stochastic_gradient_descent_steps()` 정의**
# - batch_size 만큼 데이터를 랜덤하게 추출하여 이를 기반으로 w1_update, w0_update를 계산

# In[8]:


def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=1000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 100000
    iter_index =0
    
    for ind in range(iters):
        np.random.seed(ind)
        
        # 전체 X, y 데이터에서 랜덤하게 batch_size만큼 데이터 추출하여 sample_X, sample_y로 저장
        stochastic_random_index = np.random.permutation(X.shape[0])
        sample_X = X[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]
        
        # 랜덤하게 batch_size만큼 추출된 데이터 기반으로 w1_update, w0_update 계산 후 업데이트
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    
    return w1, w0


# In[9]:


w1, w0 = stochastic_gradient_descent_steps(X, y, iters=1000)
print("w1:",round(w1[0,0],3),"w0:",round(w0[0,0],3))
y_pred1 = w1[0,0] * X + w0
print('Stochastic Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))


# **확률적 경사하강법으로 계산된 회귀계수를 적용한 회귀직선 시각화**

# In[10]:


plt.scatter(X, y)
plt.plot(X, y_pred, 'r-.')
plt.plot(X, y_pred1, 'g:')


# ### 변수가 M개 $(X_1, X_2, ... , X_{100})$인 경우 회귀계수는?

# - 선형대수를 이용하여 회귀계수 예측 가능
# ![image.png](attachment:image.png)

# # 사이킷런 LinearRegression을 이용한 보스턴 주택 가격 예측

# ### 사이킷런의 `linear_model` 모듈
# - 매우 다양한 종류의 선형 기반 회귀를 클래스로 구현
# - http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ## LinearRegression 클래스 : Ordinary Least Squares

# ### LinearRegression 클래스
# - RSS(Residual SUm of Squares)를 최소화하는 OLS(Ordinary Least Squares) 추정 방식으로 구현한 클래스
# - fit() 메서드로 X, y 배열을 입력받으면 회귀계수 W를 coef_ , intercept_ 속성에 저장
#     - coef_ : 회귀계수 $w_i$ 값들
#     - intercept_ : intercept 값 (절편)
class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
# - 입력 파라미터
# ![image.png](attachment:image.png)

# ### 다중공선성(muliti-collinearity) 문제
# - OLS 기반의 회귀계수 계산은 입력 피처의 독립성에 많은 영향을 받음
# - 피처 간의 상관관계가 매우 높은 경우 분산이 매우 커져서 오류에 민감해짐
# 
# ### 다중공선성 해결방안
# - 독립적인 중요한 피처만 남기고 제거하거나 규제를 적용
# - 매우 많은 피처가 다중공선성 문제를 가지고 있는 경우 PCA를 통해 차원축소를 수행

# ### 회귀 평가 지표

# MSE (Mean Squared Error) : 실제값과 예측값의 차이를 제곱해서 평균한 것  
#     
# RMSE (Root Mean Squared Error) : 
# - MSE 값은 오류의 제곱을 구하므로  
# - 실제 오류 평균보다 더 커지는 특성이 있으므로 MSE에 루트를 쒸운 것
# - 사이킷런에서는 제공하지 않기 때문에
# - MSE에 제곱근을 씌워서 계산하는 함수 직접 만들어서 사용
# 
# R제곱 : 분산 기반으로 예측 성능 평가. 
# - 실제값의 분산 대비 예측값의 분산 비율을 지료로 하며
# - 1에 가까울수록 예측 정확도가 높음

# ![image.png](attachment:image.png)

# - 사이킷런은 RMSE를 제공하지 않으므로, MSE에 제곱근을 적용하여 계산하는 함수를 직접 만듬

# **사이킷런의 API 및 cross_val_score()나 GridSearchCV에 평가 시 사용되는 Scoring 파라미터 적용 값**
# ![image.png](attachment:image.png)

# **cross_val_score()나 GridSearchCV와 같은 Scoring 함수에 회귀 평가 지표를 적용할 때 유의점**
# 
# : MAE, MSE의 scoring 파라미터 값에 'neg'가 붙어 있음
# - 이는 Negative를 의미하는데, 사이킷런의 Scoring은 클수록 좋은 평가결과로 자동 평가하기 때문에 -를 붙임
# - 즉, MAE, MSE가 작을수록 좋은 평가이므로, -를 붙여 값이 크게 함으로써 좋은 평가 결과로 반영하기 위함

# ## 선형 회귀 사이킷런 사용법

# **앞에서 생성한 가상 데이터(y=6+4X+noise)에 대한 LinearRegression 적용**

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


np.random.seed(0)

# y = 4X + 6 식을 근사(w1=4, w0=6). random 값은 Noise를 위해 만듬
X = 2 * np.random.rand(100,1)
y = 6 + 4 * X + np.random.randn(100,1)

# X, y 데이터 셋 scatter plot으로 시각화
plt.scatter(X, y)


# In[12]:


# 단순 선형 회귀 분석
# LinearRegression 알고리즘으로 학습
# X : 피처값
# y : 타겟값
from sklearn.linear_model import LinearRegression

line_fitter = LinearRegression() # 객체 생성
line_fitter.fit(X, y) # 학습


# In[13]:


# X가 1.5248의 타겟값 예측
y_pred = line_fitter.predict(np.array([[1.5248]]))
y_pred


# In[14]:


# 회귀 계수 : coef_ 속성
line_fitter.coef_


# In[15]:


# 절편 확인 : y = 4X + 6 (절편 : 6)
line_fitter.intercept_


# In[16]:


# 단순 선형회귀 시각화
plt.plot(X, y, 'o')
plt.plot(X,line_fitter.predict(X))
plt.show()


# ## LinearRegression을 이용한 보스턴 주택 가격 예측

# - 사이킷런에 내장된 보스턴 주택가격 데이터 **`load_boston()`** 을 사용
# ![image.png](attachment:image.png)

# ### 주택 가격 데이터 세트 로드 및 데이터 프레임으로 변환 

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston
get_ipython().run_line_magic('matplotlib', 'inline')

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(data=boston.data, columns=boston.feature_names)
# print(bostonDF.shape)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE']= boston.target
bostonDF.head()


# * CRIM: 지역별 범죄 발생률  
# * ZN: 25,000평방피트를 초과하는 거주 지역의 비율
# * NDUS: 비상업 지역 넓이 비율
# * CHAS: 찰스강에 대한 더미 변수(강의 경계에 위치한 경우는 1, 아니면 0)
# * NOX: 일산화질소 농도
# * RM: 거주할 수 있는 방 개수
# * AGE: 1940년 이전에 건축된 소유 주택의 비율
# * DIS: 5개 주요 고용센터까지의 가중 거리
# * RAD: 고속도로 접근 용이도
# * TAX: 10,000달러당 재산세율
# * PTRATIO: 지역의 교사와 학생 수 비율
# * B: 지역의 흑인 거주 비율
# * LSTAT: 하위 계층의 비율
# * MEDV: 본인 소유의 주택 가격(중앙값)

# * 각 컬럼별로 주택가격에 미치는 영향도를 조사

# In[20]:


# 2개의 행과 4개의 열을 가진 subplots를 이용. axs는 4x2개의 ax를 가짐.
fig, axs = plt.subplots(figsize=(16,8) , ncols=4 , nrows=2)

lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']

for i , feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4

    # Seaborn의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y='PRICE', data=bostonDF, ax=axs[row][col])


# In[8]:


# 각 피처에 대한 분포도 확인

fig, axs = plt.subplots(figsize=(16, 8) , ncols=4 , nrows=2)
lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']

for i , feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4
    sns.histplot(bostonDF[feature], ax=axs[row][col])


# **학습과 테스트 데이터 세트로 분리하고 학습/예측/평가 수행**

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target,
                                                       test_size=0.3, random_state=156)

# Linear Regression OLS로 학습/예측/평가 수행 
lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)

print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))


# **회귀식 추정 : 회귀계수**

# In[10]:


print('절편 값:',lr.intercept_)
print('회귀 계수값:', np.round(lr.coef_, 1))


# In[11]:


# 회귀 계수를 큰 값 순으로 정렬하기 위해 Series로 생성. index가 컬럼명에 유의
coeff = pd.Series(data=np.round(lr.coef_, 1), index=X_data.columns )
coeff.sort_values(ascending=False)


# - RM이 양의 값으로 회귀계수가 가장 크며, NOX의 회귀계수는 음의 값으로 매우 큼

# **교차 검증으로 MSE와 RMSE 측정**

# In[22]:


from sklearn.model_selection import cross_val_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)
lr = LinearRegression()

# cross_val_score( )로 5 Fold 셋으로 MSE 를 구한 뒤 이를 기반으로 다시  RMSE 구함. 
neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# cross_val_score(scoring="neg_mean_squared_error")로 반환된 값은 모두 음수 
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))

