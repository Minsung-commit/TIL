#!/usr/bin/env python
# coding: utf-8

# # 규제 선형 모델(Regularized Linear Models)

# ## 규제 선형 모델의 개요

# : 회귀모델은 적절히 데이터에 적합하면서도 회귀 계수가 기하급수적으로 커지는 것을 제어

# **선형 모델의 RSS 최소화하는 비용함수의 한계점**
# - 실제값과 예측값 차이를 최소화하는 것만 고려함에 따라
# - 학습 데이터에 지나치게 맞추게 되고 회귀계수가 쉽게 커짐
# - 이러한 경우 변동성이 심해져서 테스트 데이터세트에서 예측성능이 저하되기 쉬움

# - 학습 데이터의 잔차 오류 값을 최소로 하는 RSS 최소화와 과적합을 방지하기 위해 회귀계수값이 커지지 않도록 하는 방법이 서로 균형을 이뤄야 함
# 
# - 회귀 계수의 크기를 제어해 과적합을 개선하려면 비용(cost) 함수의 목표는
# $ RSS(W) + alpha*||W||_2^2$ 를 최소화하는 것으로 변경해야 함
# 
# ![image.png](attachment:image.png)

# $ 비용함수목표 = Min(RSS(W)) + alpha*||W||_2^2 $
# 
# - $alpha$=0인 경우 : $W$가 커도 $alpha*||W||_2^2$가 0이 되어 비용함수는 $Min(RSS(W))$
# 
# - $alpha$=무한대인 경우 : $alpha*||W||_2^2$가 무한대가 되어 비용함수는 $W$를 0에 가깝게 최소확해야 함
# 
# ![image-2.png](attachment:image-2.png)

# ## 릿지회귀(Ridge Regression)

# - L2 Norm Reqularization
# - 과적합을 피하고 일반화 성능을 강화하는 방법
# 
# $ 비용함수목표 = Min(RSS(W)) + alpha*||W||_2^2 $
# 
# ![image.png](attachment:image.png)
# 
# https://rk1993.tistory.com/entry/Ridge-regression%EC%99%80-Lasso-regression-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0

# ### 릿지 회귀의 특징
# - 변수 간 상관관계가 높은 상황(다중공선성)에서 좋은 예측 성능
# - 회귀계수의 크기가 큰 변수를 우선적으로 줄이는 경향이 있음
# - 변수 선택 불가능
# - 제약범위가 원의 형태
# 
# ![image.png](attachment:image.png)

# ### 사이킷런에서 릿지회귀 클래스 `Rdige`
from sklearn.linear_model import Ridge

Ridge(alpha=1.0, * , fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
# In[2]:


# 보스톤 주택가격 데이터 세트 이용
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)

# 릿지회귀로 예측하고, 예측 성능을 cross_cal_score() 평가
ridge = Ridge(alpha = 10)
neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 3))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores,3))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))


# => 규제가 없는 LinearRegression의 RMSE 평균 5.829보다 더 작은 값으로 더 좋은 성능을 보여줌

# **Ridge에서 alpha값을 0 , 0.1 , 1 , 10 , 100 으로 변경하면서 RMSE 측정**

# In[3]:


# Ridge에 사용될 alpha 파라미터의 값들을 정의
alphas = [0 , 0.1 , 1 , 10 , 100]

# alphas list 값을 iteration하면서 alpha에 따른 평균 rmse 구함.
for alpha in alphas :
    ridge = Ridge(alpha = alpha)
    
    #cross_val_score를 이용하여 5 fold의 평균 RMSE 계산
    neg_mse_scores = cross_val_score(ridge, X_data, y_target, 
                                     scoring="neg_mean_squared_error", 
                                     cv = 5)
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    print('alpha {0} 일 때 5 folds 의 평균 RMSE : {1:.3f} '.format(alpha,avg_rmse))


# **각 alpha에 따른 회귀 계수 값을 시각화**
# - 각 alpha값 별로 plt.subplots로 맷플롯립 축 생성

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 각 alpha에 따른 회귀 계수 값을 시각화하기 위해 5개의 열로 된 맷플롯립 축 생성  
fig , axs = plt.subplots(figsize=(18,6) , nrows=1 , ncols=5)
# 각 alpha에 따른 회귀 계수 값을 데이터로 저장하기 위한 DataFrame 생성  
coeff_df = pd.DataFrame()

# alphas 리스트 값을 차례로 입력해 회귀 계수 값 시각화 및 데이터 저장. pos는 axis의 위치 지정
for pos , alpha in enumerate(alphas) :
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_data , y_target)
    
    # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가.  
    coeff = pd.Series(data=ridge.coef_ , index=X_data.columns )
    colname='alpha:'+str(alpha)
    coeff_df[colname] = coeff
    
    # 막대 그래프로 각 alpha 값에서의 회귀 계수를 시각화. 회귀 계수값이 높은 순으로 표현
    coeff = coeff.sort_values(ascending=False)
    
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3,6)
    sns.barplot(x=coeff.values , y=coeff.index, ax=axs[pos])

# for 문 바깥에서 맷플롯립의 show 호출 및 alpha에 따른 피처별 회귀 계수를 DataFrame으로 표시
plt.show()


# **alpha 값에 따른 컬럼별 회귀계수 출력**

# In[5]:


ridge_alphas = [0 , 0.1 , 1 , 10 , 100]
sort_column = 'alpha:'+str(ridge_alphas[0])
coeff_df.sort_values(by=sort_column, ascending=False)


# ## 라쏘 회귀(Lasso Regression)

# - L1 Norm Reqularization
# 
# $ 비용함수목표 = Min(RSS(W)) + alpha*||W||_1 $
# 
# ![image-3.png](attachment:image-3.png)
# 
# https://rk1993.tistory.com/entry/Ridge-regression%EC%99%80-Lasso-regression-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0

# ### 라쏘 회귀의 특징
# - 제약 범위가 각진 형태
# - 회귀계수 일부가 0이 되어 변수 선택 기법으로 활용
# - 변수 간 상관관계 높은 상황에서 Ridge에 비해 상대적으로 예측 성능이 떨어짐
# 
# ![image-2.png](attachment:image-2.png)

# ### 사이킷런의 Lasso 회귀를 위한 클래스
from sklearn.linear_model import Lasso

Lasso(alpha=1.0, *, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
# ### 규제가 있는 회귀모델 적용 함수 get_linear_reg_eval() 작성
# - 매개변수로 규제회귀 Ridge, Lasso, ElasticNet을 지정하여 해당 규제 모델을 학습하고 에측성능 출력

# In[13]:


from sklearn.linear_model import Lasso, ElasticNet

# alpha값에 따른 회귀 모델의 폴드 평균 RMSE를 출력하고 회귀 계수값들을 DataFrame으로 반환 
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None, 
                        verbose=True, return_coeff=True):
    coeff_df = pd.DataFrame()
    if verbose : print('####### ', model_name , '#######')
    for param in params:
        if model_name =='Ridge': model = Ridge(alpha=param)
        elif model_name =='Lasso': model = Lasso(alpha=param)
        elif model_name =='ElasticNet': model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, 
                                             y_target_n, scoring="neg_mean_squared_error", cv = 5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print('alpha {0}일 때 5 폴드 세트의 평균 RMSE: {1:.3f} '.format(param, avg_rmse))
        # cross_val_score는 evaluation metric만 반환하므로 모델을 다시 학습하여 회귀 계수 추출
        
        model.fit(X_data_n , y_target_n)
        if return_coeff:
            # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가. 
            coeff = pd.Series(data=model.coef_ , index=X_data_n.columns )
            colname='alpha:'+str(param)
            coeff_df[colname] = coeff
    
    return coeff_df
# end of get_linear_regre_eval


# **alpha값을 0.07, 0.1, 0.5, 1.3으로 지정한 경우 라쏘모델의 성능 평가**

# In[14]:


# 라쏘에 사용될 alpha 파라미터의 값들을 정의하고 get_linear_reg_eval() 함수 호출
lasso_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df =get_linear_reg_eval('Lasso', params=lasso_alphas, 
                                    X_data_n=X_data, y_target_n=y_target)


# => alpha가 0.07일때 가장 좋은 평균 RMSE를 보여주나, 릿지 경우보다 약간 떨어짐

# In[15]:


print(lasso_alphas)


# **alpha값에 따른 피처별 회귀 계수**

# In[96]:


# 반환된 coeff_lasso_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame출력
sort_column = 'alpha:'+str(lasso_alphas[0])
coeff_lasso_df.sort_values(by=sort_column, ascending=False)


# => alpha가 증가함에 따라 일부 회귀계수는 0이 됨

# ## 엘라스틱넷 회귀(ElasticNet Regression)

# - L1과 L2 규제를 결합한 회귀
# 
# 
# - 비용함수목표 = $RSS(W) + alpha2*||W||_2^2 + alpha1*||W||_1 $
# 
# 
# - 라쏘 회귀의 단점을 완화하기 위해 L2규제를 라쏘회귀에 추가한 것
#     - 라쏘회귀는 서로 상관관계가 높은 변수(피처)들의 경우 중요 변수만 선택하고 다른 변수의 회귀계수는 0으로 만들어 회귀계수의 값이 급격히 변동하는 특징을 완화시키기 위함
# 
# 
# - 수행시간이 오래걸리는 단점이 있음
# 
# ![image.png](attachment:image.png)

# ### 사이킷런의 엘라스틱넷 회귀 클래스 `ElasticNet`
ElasticNet(alpha=1.0, * , l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
# **ElasticNet의 주요 파라미터**
# 
# 엘라스틱넷의 규제는 $a*L1 + b*L2$로 정의
# 
# (여기서 a는 L1규제의 alpha값, b는 L2규제의 alpha 값)
# 
# - alpha : a+b
# - l1_ratio : a/(a+b)
#     - l1_ratio가 0이면 a가 0이므로 L2 규제와 동일
#     - l1_ratio가 1이면 b가 0이므로 L1 규제와 동일

# **엘라스틱넷에 사용될 alpha값을 변화시키면서 성능평가**

# In[16]:


# 엘라스틱넷에 사용될 alpha 파라미터의 값들을 정의
# 앞에서 작성한 get_linear_reg_eval() 함수 호출
# 이 함수에서 l1_ratio는 0.7로 고정되어 있음
# alpha값의 변화만 보기 위함

elastic_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df =get_linear_reg_eval('ElasticNet', params=elastic_alphas,
                                      X_data_n=X_data, y_target_n=y_target)


# => alpha가 0.5일때 가장 좋은 예측성능을 보임

# In[17]:


# 반환된 coeff_elastic_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame출력
sort_column = 'alpha:'+str(elastic_alphas[0])
coeff_elastic_df.sort_values(by=sort_column, ascending=False)


# => 회귀계수들이 라쏘회귀의 경우보다 상대적으로 0이 되는 값이 적음

# ## 선형 회귀 모델을 위한 데이터 변환

# - 선형회귀모델의 가정에 의해 타깃값(y)이 정규분포 형태를 많이 선호함
#     - 가정 : 선형성, 정규성, 독립성, 등분산성
#     
# - 선형회귀모델을 적용하기 위해 데이터에 대한 스케일링/정규화 작업을 수행하는 것이 일반적
#     - 중요 피처들이나 타깃값의 분포도가 심하게 비대칭인 경우

# 1. StandardScaler 클래스
# 
# 
# 2. MinMaxScaler 클래스
# 
# 
# 3. 스케일링/정규화를 수행한 데이터세트에 다시 다항특성을 적용하여 변환
#     - 스케일링/정규화를 했으나 예측성능에 향상이 없을 경우
#     
#     
# 4. 비대칭분포(오른쪽으로 꼬리가 긴 분포)의 경우 로그 변환

# **데이터변환을 위한 함수 get_scaled_data() 작성**

# In[18]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# method는 표준 정규 분포 변환(Standard), 최대값/최소값 정규화(MinMax), 로그변환(Log) 결정
# p_degree는 다향식 특성을 추가할 때 적용. p_degree는 2이상 부여하지 않음. 
def get_scaled_data(method='None', p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log':
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(degree=p_degree, 
                                         include_bias=False).fit_transform(scaled_data)
    
    return scaled_data


# In[19]:


# 보스톤 주택가격데이터

# Ridge의 alpha값을 다르게 적용하고 다양한 데이터 변환방법에 따른 RMSE 추출. 
alphas = [0.1, 1, 10, 100]

# 변환 방법은 모두 6개, 원본 그대로, 표준정규분포, 표준정규분포+다항식 특성
# 최대/최소 정규화, 최대/최소 정규화+다항식 특성, 로그변환 

scale_methods=[(None, None), ('Standard', None), ('Standard', 2), 
               ('MinMax', None), ('MinMax', 2), ('Log', None)]

for scale_method in scale_methods:
    X_data_scaled = get_scaled_data(method=scale_method[0], p_degree=scale_method[1], 
                                    input_data=X_data)
    print(X_data_scaled.shape, X_data.shape)
    print('\n## 변환 유형:{0}, Polynomial Degree:{1}'.format(scale_method[0], scale_method[1]))
    get_linear_reg_eval('Ridge', params=alphas, X_data_n=X_data_scaled, 
                        y_target_n=y_target, verbose=False, return_coeff=False)

