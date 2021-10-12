#!/usr/bin/env python
# coding: utf-8

# # 앙상블 학습

# ## 앙상블 학습 (Ensemble Learning) 개요
# 
# ### 앙상블 학습을 통한 분류
# - 여러 개의 분류기(Classifier)을 사용해서 예측 결합함으로써 보다 정확한 최종 예측을 도출하는 기법
# - 단일 분류기 사용 때보다 신뢰성이 높은 예측값을 얻을 수 있음
# - 쉽고 편하면서도 강력한 성능 보유
# - 대부분의 정형 데이터 분류 시 뛰어난 성능을 나타냄
# - 이미지, 영상, 음성 등의 비정형 데이터 분류 : 딥러닝 성능 뛰어남

# ### 대표적인 앙상블 알고리즘
# - 랜덤 포레스트(Random Forrest)
# - 그레디언트 부스팅(Gradient Boosting)

# ### 앙상블 알고리즘 변화
# - 뛰어난 성능, 쉬운 사용, 다양한 활용도로 인해 많이 애용되었고
# - 부스팅 계열의 앙상블 알고리즘의 인기와 강세가 계속 이어져
# - 기존의 그레디언트 부스팅을 뛰어넘는 새로운 알고리즘 가속화
# 
# **최신 앙상블 알고리즘**
# - XGBoost
# - LightBGM : XGBoost와 예측 성능 유사하면서도 수행 속도 훨씬 빠름
# - Stacking : 여러 가지 모델의 결과를 기반으로 메타 모델 수립
# 
# 
# XGBoost, LightBGM과 같은 최신 앙상블 알고리즘 한두 개만 잘 알고 있어도
# 정형 데이터의 분류 또는 회귀 분야에서 예측 성능이 매우 뛰어난 모델을 쉽게 만들 수 있음

# ## 앙상블 학습 유형
# - 보팅(Voting)
# - 배깅(Bagging)
# - 부스팅(Boosting)
# - 스태킹(Stacking)

# 보팅(Voting) : 여러 분류기가 투표를 통해 최종 예측 결과를 결정하는 방식
# - 일반적으로 서로 다른 알고리즘을 가진 분류기를 결합
# 
# 배깅(Bagging) : 보팅과 동일하게 여러 분류기가 투표를 통해 최종 예측 결과를 결정하는 방식
# - 각각의 분류기가 모두 같은 유형의 알고리즘 기반이지만, 
# - 샘플링을 서로 다르게 하면서 학습 수행
# - 대표적인 배깅 방식 : 랜덤 포레스트 알고리즘

# ![image-2.png](attachment:image-2.png)

# 보팅 분류기 도식화
# - 선형회귀, K최근접 이웃, 서포트 벡터 머신 3개의 ML 알고리즘이
# - 같은 데이터 세트에 대해 학습하고 예측한 결과를 가지고
# - 보팅을 통해 최종 예측 결과를 선정
# 
# 배깅 분류기 도식화
# - 단일 ML 알고리즘(결정트리)만 사용해서
# - 여러 분류기가 각각의 샘플링된 데이터 세트에 대해 학습하고 개별 예측한 결과를
# - 보팅을 통해 최종 예측 결과 선정

# 샘플링 방식 : 부트 스트래핑 분할 방식
# - 개별 Classifier에게 데이터를 샘플링해서 추출하는 방식
# - 각 샘플링된 데이터 내에는 중복 데이터 포함
# - (교차 검증에서는 데이터 세트 간에 중첩 허용하지 않음)

# ![image.png](attachment:image.png)
# https://swalloow.github.io/bagging-boosting/

# ### 부스팅(Boosting)
# - 여러 개의 분류기가 순차적으로 학습 수행하되
# - 앞에서 학습한 분류기가 예측이 틀린 데이터에 대해서는 올바르게 예측할 수 있도록
# - 다음 분류기에게는 가중치(weight)를 부여하면서 
# - 학습과 예측을 진행하는 방식
# - 예측 성능이 뛰어나 앙상블 학습 주도
# - boost : 밀어 올림
#     
# **대표적인 부스팅 모듈**
# - Gradient Boost
# - XGBost(eXtra Gradient Boost)
# - LightGBM(Light Gradient Boost)

# ### 스태킹
# - 여러 가지 다른 모델의 예측 결과값을 다시 학습 데이터로 만들어로
# - 다른 모델(메타 모델)로 재학습시켜 결과를 예측하는 방식

# ### 보팅 유형 
# - 하드 보팅 (Hard Voting)
# - 소프트 보팅 (Soft Voting)
# 
# 하드 보팅 (Hard Voting)
# - 다수결 원칙과 유사
# - 예측한 결과값들 중에서 
# - 다수의 분류기가 결정한 예측값을
# - 최종 보팅 결과값으로 선정
# 
# 소프트 보팅 (Soft Voting)
# - 분류기들의 레이블 값 결정 확률을 평균내서
# - 확률이 가장 높은 레이블 값을
# - 최종 보팅 결과값으로 선정
# - 일반적으로 소프트 보팅이 예측 성능이 좋아서 더 많이 사용

# ![image-2.png](attachment:image-2.png)

# 하드 보팅 도식화
# - Classifier 1, 2, 3, 4번 4개로 구성
# - 분류기 1, 3, 4번 예측 : 레이블 값 1로 예측
# - 분류기 2번 예측 : 2로 예측
# - 다수결 원칙에 따라서 최종 예측은 레이블 값 1
# 
# 소프트 보팅  도식화
# - 레이블 값1과 레이블 값2에 대한 분류기 별 예측 확률
# - 1번 : 0.7, 0.3
# - 2번 : 0.2, 0.8
# - 3번 : 0.8, 0.2
# - 4번 : 0.5, 0.1
# - 레이블 값 1예 대한 예측 확률 평균 : 0.64 
# - 레이블 값 2예 대한 예측 확률 평균 : 0.35 
# - 최종 레이블 값 1로 최종 보팅

# ## Voting Classifier

# ### 보팅 방식의 앙상블 예제 : 위스콘신 유방암 데이터 세트 예측 분석  
#         
# **위스콘신 유방암 데이터 세트**
# - 유방암의 악성종양, 양성종양 여부를 결정하는 이진 분류 데이터 세트
# - 종양의 크기, 모양 등의 형태와 관련한 많은 피처 포함
# - 사이킷런의 보팅 양식의 앙상블을 구현한 VotingClassifier 클래스를 이용해서 보팅 분류기 생성  
# - `load_breast_cancer()` 함수를 통해 위스콘신 유방암 데이터 세트 생성
# - 로지스틱 회귀와 KNN 기반으로 소프트 보팅 방식으로 보팅 분류기 생성

# ### 위스콘신 유방암 데이터 로드

# In[3]:


import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

# 피처 확인
data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_df.head()


# In[2]:


cancer.data.shape


# **VotingClassifier로 개별모델은 로지스틱 회귀와 KNN을 보팅방식으로 결합하고 성능 비교**

# VotingClassifier 클래스의 주요 생성 인자
# - estimators : 리스트 값으로 보팅에 사용될 여러 개의 Classifier 객체들을 튜플 형식으로 입력 받음
# - voting : 보팅 방식 - hard/soft (디폴트 : hard) 

# In[13]:


# 개별 모델은 로지스틱 회귀와 KNN 임. 
lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=8)

# 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기 

# estimators 복수 : 리스트형태
# lr_clf 이름을 'LR'로
# knn_clf 이름을 'KNN'으로 
# 보팅 방식 : 디폴트 hard

vo_clf = VotingClassifier(estimators=[('LR', lr_clf),('KNN', knn_clf)], voting='soft')

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    test_size=0.2, random_state=156)


# In[14]:


# VotingClassifier 학습/예측/평가. 
# 개별 모델들이 다 학습하고 예측한 결과로 평가
vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
print('보팅을 통한 분류기 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))


# In[10]:


# 로지스틱 회귀와 KNN 각개별 모델의 학습/예측/평가.
classifiers = [lr_clf, knn_clf]

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    class_name = classifier.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))


# - 보팅 분류기가 정확도가 조금 높게 나타났는데
# 보팅으로 여러 개의 기반 분류기를 결합한다고 해서 무조건 예측 성능이 더 향상되지는 않음
# - 보팅, 배깅, 부스팅 등의 앙상블 방법은
# 전반적으로 다른 단일 ML 알고리즘 보다 뛰어난 예측 성능을 가지는 경우가 많음
# 
# - 고정된 데이터 세트에서 단일 ML 알고리즘이 뛰어난 성능을 발휘하더라도
# 현실 세계는 다양한 변수와 예측이 어려운 규칙으로 구성되어있기 때문에
# 다양한 관점을 가진 알고리즘이 서로 결합해 더 나은 성능을 실제 환경에서 끌어낼 수 있음
#     
# 저자 설명    
# - 살짝 만들어진 결과 냄새가 난다. 그런쪽으로 좀 유도도 하긴 했음
# - 모든 걸 다 합친다고 개별 모델보다 무조건 좋아지라는 법은 없음
# - 똑똑한 하나가 여러 개를 합친 것 보다 좋을 수도 있고, 그렇지 않을 수도 있고
# - 보팅을 통해서 개별 모델들을 합치면 좋아질 수 있는 가능성이 있다 정도로 이해하면 됨

# # 랜덤 포레스트(Random Forest)

# ### 배깅(Bagging)
# - 보팅과는 다르게 같은 알고리즘으로 여러 개의 분류기를 만들어서 보팅으로 최종 결정하는 알고리즘
# - 대표적 배깅 알고리즘 : 랜덤 포레스트
#     
# ### 랜덤 포레스트(Random Forest)
# - 다재 다능한 알고리즘
# - 앙상블 알고리즘 중 수행 속도가 빠르고
# - 다양한 영역에서 높은 예측 성능을 보임
# - 기반 알고리즘은 결정 트리
# - 결정 트리의 쉽고 직관적인 장점을 그대로 채택
# - (대부분의 부스팅 기반의 다양한 알고리즘 역시 
# - 결정 트리 알고리즘을 기반 알고리즘으로 채택)
# 
# ### 랜덤 포레스트의 예측 결정 방식
# - 여러 개의 결정 트리 분류기가
# - 전체 데이터에서 배깅 방식으로 각자의 데이터를 샘플링하여
# - 개별적으로 학습을 수행한 뒤
# - 최정적으로 모든 분류기가 보팅을 통해 예측 결정

# ![image-2.png](attachment:image-2.png)

# ### 랜덤 포레스트에서의 부트스트래핑 샘플링 방식
# 
# **부트스트래핑(bootstrapping) 분할 방식**
# - 개별 Classifier에게 데이터를 샘플링해서 추출하는 방식
# - 각 샘플링된 데이터 내에는 중복 데이터 포함
# 
# **랜덤 포레스트 부트 부트 스트래핑 분할**
# - 개별적인 분류기의 기반 알고리즘은 결정 트리
# - 개별 트리가 학습하는 데이터 세트는 전체 데이터에서 일부가 중복되게 샘플링된 데이터 세트
# - Subset 데이터는 이러한 부트 스트래핑으로 데이터가 임의로 만들어짐
# - Subset 데이터 건수는 전체 데이터 건수와 동일하지만 개별 데이터가 중복되어 만들어짐
# 
# 
# - 예 : 원본 데이터 건수가 10개인 학습 데이터 세트
#     - 랜덤 포레스트를 3개의 결정 트리 기반으로 학습하려고
#     - n_estimators = 3으로 하이퍼 파라미터를 부여한 경우

# ![image-2.png](attachment:image-2.png)

# ## 랜덤 포레스트 예제
# - 앞의 사용자 행동 인식 데이터 세트를 
# - 사이킷런의 RandomForestClassifier 클래스를 이용해 예측 수행

# ### 결정 트리에서 사용한 사용자 행동 인지 데이터 세트 로딩

# In[15]:


# 앞에서 결정트리 예제에서 작성했음 (복사해서 사용)

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


# In[16]:


# 앞에서 작성했음 (복사해서 사용)

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
    y_train = pd.read_csv('data/human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])
    y_test = pd.read_csv('data/human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])
    
    # 로드된 학습/테스트용 DataFrame을 모두 반환 
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_human_dataset()


# ### 학습/테스트 데이터로 분리하고 랜덤 포레스트로 학습/예측/평가

# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 결정 트리에서 사용한 get_human_dataset( )을 이용해 학습/테스트용 DataFrame 반환
X_train, X_test, y_train, y_test = get_human_dataset()

# 랜덤 포레스트 학습 및 별도의 테스트 셋으로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train , y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test , pred)
print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy))
# 수행속도 괜찮게 나왔지만 좋은 편은 아니다


# ### 트리 기반의 앙상블 알고리즘의 단점
# - 하이퍼파라미터가 너무 많고
# - 그로 인해서 튜닝을 위한 시간이 많이 소모된다는 것
# - 또한 많은 시간을 소모했으나 튜닝 후 예측 성능이 크게 향상되는 경우가 많지 않음
# - 트리 기반 자체의 하이퍼파라미터가 원래 많으며
#     - 배깅, 부스팅, 학습, 정규화 등을 위한 하이퍼 파라미터까지 추가되므로
#     - 일반적으로 다른 ML 알고리즘에 비해 많을 수 밖에 없음
# - 그나마 랜덤 포레스트가 적은 편에 속하는데,
# - 결정 트리에서 사용되는 하이퍼 파라미터와 같은 파라미터가 대부분이기 때문

# ### GridSearchCV 로 교차검증 및 하이퍼 파라미터 튜닝

# **GridSearchCV 로 교차검증 및 하이퍼 파라미터 튜닝**
# - 앞의 사용자 행동 데이터 세트 그대로 사용
# - 튜닝 시간을 절약하기 위해 
#     - n_estimators=100
#     - cv=2
#     
# 예제 수행 시간 오래 걸림
# - 멀티 코어 환경에서는 빠르게 학습이 가능
# - 그래서 그래디언트 부스팅보다 예측 성능이 약간 떨어지더라도
# - 랜덤 포레스트로 일단 기반 모델을 먼저 구축하는 경우가 많음
# - 멀티 코어 환경에서는 n_jobs=-1로 추가하면 모든 CPU 코어 이용해 학습

# n_estimators : 결정 트리의 개수. 디폴트 10
# - 많이 설정할수록 좋은 성능을 기대할 수 있지만
# - 계속 증가시킨다고 무조건 향샹되는 것은 아님
# - 또 증가시킬수록 학습 수행 시간이 오래 걸림

# In[18]:


from sklearn.model_selection import GridSearchCV

# 테스트해 볼 데이터를 많이 넣으면 기하급수적으로 늘어난다
params = {
    'n_estimators':[100], # 1차적으로 100으로 줄이고,나중에 최적화 되면 늘려서 최종적으로 예측 수행
    'max_depth' : [6, 8, 10, 12], 
    'min_samples_leaf' : [8, 12, 18 ],
    'min_samples_split' : [8, 16, 20]  # 4x3x3 : 36번
}

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1) 

# n_jobs=-1 : 전체 cpu 콜을 다 활용하라는 거고
# n_jobs=-1 를 사용하면 개인 pc가 굉장히 많은 수행 성능을 잡아 먹기 때문에 느려짐
# 수행하면 한 2분 걸림

grid_cv = GridSearchCV(rf_clf , param_grid=params , cv=2, n_jobs=-1 ) 
# cv 2개 : 너무 많이 하면 실행시간이 오래걸리니까 (총 72번 수행 : 36 x 2)

grid_cv.fit(X_train , y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))

# 결과
# 출력된 최적 하이퍼 파라미터일 때 최고 예측 정확도: 91.8 %


# ### 튜닝된 하이퍼 파라미터로 재 학습 및 예측/평가

# In[19]:


# GridSearchCV로 찾은 최적의 하이퍼 파라미터를
# 랜덤 포레스트에 적용해서 예측 수행

# 이번에는 n_estimators=300으로 늘림
# 위 결과의 최적 하이퍼 파라미터들을 다 입력해서 
# RandomForestClassifier 초기화시키고
# 예측 성능 측정

rf_clf1 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8,
                                 min_samples_split=8, random_state=0)
rf_clf1.fit(X_train , y_train)
pred = rf_clf1.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test , pred)))


# ### 개별 feature들의 중요도 시각화

# 결정 트리에서처럼 feature_importances_ 속성을 이용해서  
# 알고리즘이 선택한 피처의 중요도를 알 수 있음  
# 
# 피처들의 중요도를 막대그래프로 시각화 

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# 앞으로 계속 중요도 시각화를 죽 계속할 건데 이 코드를 계속 비슷하게 사용한다고 생각하면 됨
ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns)
# sort_values() 쉽게 하기 위해서 시리즈로 만들고, 
# 최고 중요도가 높은 20개 피처들만 추출
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
# x축은 중요도 값, y축은 ftr_top20 시리즈의 index
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()


# # GBM(Gradient Boosting Machine)

# ### 부스팅(Boosting)
# - 여러 개의 약한 학습기(weak learner)를 순차적으로 학습-예측하면서
# - 잘못 예측된 데이터에 가중치(weight) 부여를 통해
# - 오류를 개선해 나가면서 학습하는 방식
# 
# ### 대표적 부스팅 알고리즘
# - AdaBoost(Adaptive Boosting) : 에이다 부스트
# - GBM(Gradient Boosting Machine) : 그래디언트 부스트

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ## GBM(Gradient Boosting Machine) : 그래디언트 부스트
# - 에이다 부스트와 유사하지만
# - 가중치 업데이터를 경사 하강법을 이용하는 것이 큰 차이
#     - 반복 수행을 통해 오류를 최소화할 수 있도록
#     - 가중치의 업데이트 값을 도출
#     - 오류값 = 실제값 - 예측값
# - 분류와 회귀 둘 다 가능

# 경사 하강법(Gradient Descent)
# - 함수의 기울기(경사)를 구하고 경사의 절대값이 낮은 쪽으로 계속 이동시켜 극값에 이를 때까지 반복시키는 것(위키백과)
# - 제시된 함수의 기울기로 최소값을 찾아내는 머신러닝 알고리즘
# - 매개변수를 반복적으로 조정해서 최소 함수값을 갖게하는 독립변수를 찾는 방법

# CART 기반 알고리즘
# - Classification And Regression Tree
# - 분류와 회귀 다 가능한 알고리즘

# ## GBM 예제
# - GBM을 이용해 사용자 행동 데이터 세트를 예측 분류 수행
# - GBM 학습하는 시간이 얼마나 걸리는지 GBM 수행 시간 측정
# - 사이킷런의 GradientBoostingClassifier 클래스 사용
# - 앞에서 작성한 get_new_feature_name_df() 함수와 get_human_dataset( ) 함수 사용

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
import time
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = get_human_dataset()

# GBM 수행 시간 측정을 위함. 시작 시간 설정.
start_time = time.time()

gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train , y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print('GBM 정확도: {0:.4f}'.format(gb_accuracy))
print("GBM 수행 시간: {0:.1f} 초 ".format(time.time() - start_time))


# In[19]:


# 결과
# 기본 하이퍼 파라미터만으로도 93.69 %의 예측 정확도로
# 앞의 튜닝된 하이퍼 파라미터로 재 학습 및 예측/평가한
# 랜덤 포레스트(91.65 %)보다 나은 예측 성능을 나타냄
# 일반적으로 GBM이 랜덤 포레스트보다 예측 성능이 조금 뛰어난 경우가 많음
# 문제 : 시간 오래 걸리고 하이퍼 파라미터 튜닝 노력도 더 필요

# GBM이 극복해야 할 중요 과제 : 수행 시간
# 사이킷런의 GradientBoostingClassifier는 
# 약한 학습기의 순차적인 예측 오류 부정을 통해 학습을 수행하므로
# 멀티 CPU 코어 시스템을 사용하더라도 병렬 처리가 지원되지 않아서
# 대용량 데이터의 경우 학습에 매우 많이 시간 필요

# 데이터가 커지면 커질수록 너무 오래 걸려서
# 하이퍼 파라미터 튜닝하기 많이 어려움

# 반면에 랜덤 포레스트의 경우 상대적으로 빠른 수행시간을 보장해주기 때문에
# 더 쉽게 예측 결과 도출 가능


#  ## GBM 하이퍼 파라미터 및 튜닝

# ### GBM의 주요 하이퍼 파라미터  
# 
# **`loss`** : 경사 하강법에서 사용할 비용 함수 지정. 기본값은 'deviance'
#     
# **`n_estimators`** : weak learner의 개수. 기본값 100
# - weak learner가 순차적으로 오류를 보정하므로
# - 개수가 많을수록 예측 성능이 일정 수준까지 좋아질 수 있음
# - 그러나 개수가 많을 수록 시간이 오래 걸림
# 
# **`learning_rate`** : GBM이 학습을 진행할 때마다 적용하는 학습률
# - weak learner가 순차적으로 오류값을 보정해 나가는 데 적용하는 계수
# - 0 ~ 1 사이의 값 지정 (기본값 0.1)
# - 작은 값을 적용하면 업데이트 되는 값이 작아져서
#     - 최소 오류 값을 찾아 예측 성능이 높아질 가능성은 높지만
#     - 많은 weak learner의 순차적인 반복 작업에 수행 시간이 올래 걸림
# - 너무 작게 설정하면 모든 weak learner의 반복이 완료되어도
#     - 최소 오류값을 찾지 못할 수도 있음
# - 반대로 큰 값을 적용하면 최소 오류값을 찾지 못하고 그냥 지차져 버려
#     - 예측 성능이 떨어질 가능성이 높아지지만 빠른 수행은 가능
# 
# **`subsample`** : weak learner가 학습에 사용하는 데이터의 샘플링 비율
# - 기본값 1 : 전체 학습 데이터를 기반으로 학습한다는 의미
# - 0.5 : 학습 데이터의 50%
# - 과적합이 염려되는 경우 1보다 작은 값으로 설정

# ### GridSearchCV 이용해서 하이퍼 파라미터 최적화

# In[ ]:


# GridSearchCV 이용해서 하이퍼 파라미터 최적화
# 사용자 행동 데이터 세트 정도의 데이터 양에
# 많은 하이퍼 파라미터로 튜닝하게 되면 시간이 상당히 오래 걸림
# 간략하게 n_estimators와 learning_rate만 적용

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100, 500],
    'learning_rate' : [ 0.05, 0.1]
}
grid_cv = GridSearchCV(gb_clf , param_grid=params , cv=2 ,verbose=1)
grid_cv.fit(X_train , y_train)
print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))

# 한 30분 걸릴 것이다

# 결과
# learning_rate이 0.05, n_estimators가 500일 때
# 2개의 교차 검증 세트에서 90.1 %의 최고 예측 정확도 도출


# In[ ]:


scores_df = pd.DataFrame(grid_cv.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score',
'split0_test_score', 'split1_test_score']]


# In[22]:


# GridSearchCV를 이용하여 최적으로 학습된 estimator로 
# 테스트 데이터 세트에 적용해서 예측 수행. 
gb_pred = grid_cv.best_estimator_.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print('GBM 정확도: {0:.4f}'.format(gb_accuracy))

# 결과
# 테스트 데이터 세트에서 약 96.06 % 정확도 도출


# - GBM은 **`수행 시간이 오래 걸린다`**는 단점이 있지만 **`과적합에도 강해서`** 예측 성능이 뛰어난 알고리즘  
# - 많은 알고리즘이 GBM을 기반으로 새롭게 만들어지고 있음
#     - 머신러닝 세계에서 가장 각광을 받는 그래디언트 부스팅 기반 ML 패키지
#         - XGBoost
#         - LightGBM
