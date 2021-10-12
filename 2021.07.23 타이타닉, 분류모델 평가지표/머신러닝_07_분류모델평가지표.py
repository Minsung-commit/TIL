#!/usr/bin/env python
# coding: utf-8

# # 모델 성능 평가

# ### 분류 모델의 평가 지표
# : 예측 대상이 범주형 데이터 경우
# - 정확도(Accuracy)
# - 재현율(Recall)
# - 정밀도(Precision)
# - F1 measure
# - G measure
# - ROC curve
# - AUC
# 
# ### 회귀 모델의 평가 지표
# : 예측 대상이 수치 데이터인 경우
# - MSE(Mean Square Error)
# - RMSE(Root Mean Square Error)
# - MAE(Mean Absolute Error)
# - MAPE(Mean Absolute Percentage Error)
# - $ R^2 $
# 
# ![image.png](attachment:image.png)
# 

# # 분류 모델의 성능 평가 지표

# ## Accuracy(정확도)

# - 실제 데이터와 예측 데이터가 얼마나 같은지를 판단하는 지표
# 
# - $ 정확도(Accuracy) =  \frac{예측 결과가 동일한 데이터 건수}{전체 예측 데이터 건수} $
# 
# 
# - 직관적으로 모델 예측 성능을 나타내는 평가 지표
# - 그러나 이진 분류의 경우 데이터의 구성에 따라 ML 모델의 성능을 왜곡할 수 있기 때문에 
#     - 정확도 수치 하나만 가지고 성능을 평가하지는 않음
# 
# 
# - 특히 정확도는 불균형한 레이블 값 분포에서 ML 모델의 성능을 판단할 경우, 적합한 지표가 아님

# ### 정확도 문제 예
# 1. 타이타닉 생존자 예측
# 2. MNIST 데이터 세트

# ### 1. 타이타닉 생존자 예측

# In[4]:


import numpy as np
from sklearn.base import BaseEstimator

# 아무런 학습을 하지 않고 성별에 따라 생존자를 예측하는 
# 단순한 Classifier 생성
# BaseEstimator 상속 받음

class MyDummyClassifier(BaseEstimator):
    # fit( ) 메소드는 아무것도 학습하지 않음.
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if X['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1
                
        return pred


# **MyDummyClassifier를 이용해 타이타닉 생존자 예측 수행**

# In[5]:


# 데이터 가공 (타이타닉 생존자 예측 시 작성)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행. 
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


# In[6]:


# 타이타닉 생존자 예측 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할. 
titanic_df = pd.read_csv('data/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test=train_test_split(X_titanic_df,
                                                  y_titanic_df,
                                                  test_size=0.2,
                                                  random_state=0)

# 위에서 생성한 Dummy Classifier를 이용하여 학습/예측/평가 수행. 

myclf = MyDummyClassifier()
myclf.fit(X_train, y_train)
mypred = myclf.predict(X_test)
print('Dummy Classifier의 정확도는: {0:.4f}'.format(accuracy_score(y_test, mypred)))


# ### 2. MNIST 데이터 세트
# * 0~9까지의 숫자 이미지의 픽셀 정보를 가지고 있고
# * 이를 기반으로 숫자 Digit을 예측하는 데 사용
# * 사이킷런의 load_digits() API를 통해 MNIST 데이터 세트 제공

# ![image-2.png](attachment:image-2.png)

# **이진 분류 문제로 변환**
# * 불균형한 데이터 세트로 변형
# * 레이블 값이 7인 것만 True, 나머지 값은 모두 False로 변환
# * True : 전체 데이터의 10%
# * False : 90%
#     
# **입력되는 모든 데이터를 False, 즉 0으로 예측하는 classifier를 이용해**
# * 정확도를 측정하면 약 90%에 가까운 예측 정확도를 나타냄

# ### 정확도 평가 지표의 맹점
# * 아무것도 하지 않고 무조건 특정한 결과로 찍어도
# * 데이터가 균일하지 않은 경우 높은 수치가 나타날 수 있음

# In[32]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# **digit 데이터 로드**

# In[11]:


# digits 확인
digits = load_digits()

digits


# In[14]:


digits.data.shape
digits.target.shape


# **7인 데이터 확인**

# In[16]:


digits.target == 7


# **7인 데이터는 1, 그외 데이터는 0으로 변환**

# In[18]:


# digits번호가 7번이면 True이고 이를 astype(int)로 1로 변환, 
# 7번이 아니면 False이고 0으로 변환. 
y = (digits.target == 7).astype(int)


# **학습 / 테스트 데이터 세트로 분리 (default = 0.25))**

# In[19]:


# 학습 / 테스트 데이터 세트로 분리 (default = 0.25)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)


# In[20]:


X_train


# In[21]:


X_test


# In[22]:


y_train


# In[23]:


y_test


# **불균형한 레이블 데이터 분포도 확인**

# In[27]:


# 불균형한 레이블 데이터 분포도 확인. 
print("y_test.shape : " , y_test.shape)
pd.Series(y_test).value_counts()


# In[33]:


# Dummy Classifier로 학습/예측/정확도 평가

fake_cl = MyFakeClassifier()
fake_cl.fit(X_train, y_train)
fakePred = fake_cl.predict(X_test)
accuracy = accuracy_score(y_test, fakePred)
print('정확도 : ', np.round((accuracy), 3))


# # Confusion Matrix (오차 행렬)

# 오차행렬 (Confusion Matrix : 혼동행렬)
# * 이진 분류의 예측 오류가 얼마인지와 더불어 어떠한 유형의 예측 오류가 발생하고 있는지를 함께 나타내는 지표
# * 학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈리고(confused) 있는지도 함께 보여주는 지표
# * 4분면 행렬에서 실제 레이블 클래스 값과 예측 레이블 클래스 값이 어떤 유형을 가지고 맵핑되는지 나타냄
# * 예측 클래스와 실제 클래스의 값 유형에 따라 TN, FP, FN, TP 형태
# * TN, FP, FN, TP 값을 다양하게 결합해 분류 모델 예측 성능의 오류가 어떤 모습으로 발생하는지 알 수 있음

# ![image-5.png](attachment:image-5.png)

# TN, FP, FN, TP는 예측 클래스와 실제 클래스의 
# * Positive 결정값(1)과 Negative 결정값(0)의 결합에 따라 결정
# * 앞 문자 T/F(True/False) : 예측값과 실제값이 '같은가/틀린가' 의미
# * 뒤 문자 N/P(Negative/Positive) : 예측 결과 값이 부정(0)/긍정(1) 의미
# * 예 : TN (True Negative) 
#     - 앞 True : 예측 클래스 값과 실제 클래스 값이 같다는 의미
#     - 뒤 Negative : 예측 값이 Negative 값이라는 의미
# 

# In[35]:


from sklearn.metrics import confusion_matrix
# MNISt의 fakepred
confusion_matrix(y_test, fakePred)


# 결과  
# [[TN, FP],  
#  [FN, TP]]
# 
# MyFakeClassifier는 load_digits()에서 target=7인지 아닌지에 따라  
# 클래스 값을 True/False 이진 분류로 변경한 데이터 세트를 사용해서  
# 무조건 Negative로 예측하는 Classifier였고  
# 테스트 데이터 세트의 클래스 값 분포는 0이 450건, 1이 45건 이었음  
# 
# * TN : 전체 450건 데이터 중 무조건 Negative 0으로 예측해서 True가 된 결과 450건
#     - 실제값/예측값 동일, Negative로 예측  
# * FP : Positive 1로 예측한 건수가 없으므로 0건
#     - 실제값/예측값 다름, Positive로 예측  
# * FN : Positive 1인 건수 45건을  Negative 0으로 예측해서 False가 된 결과 45건
#     - 실제값/예측값 다름, Negative로 예측  
# * TP : Positive 1로 예측한 건수가 없으므로 0건
#     - 실제값/예측값 동일, Positive로 예측  

# ![image-2.png](attachment:image-2.png)

# **TN, FP, FN, TP 값은 Classifier 성능의 여러 면모를 판단할 수 있는 기반 정보 제공**
# - 이 값을 조합해 Classifier의 성능을 측정할 수 있는 주요 지표인 정확도(Accuracy), 정밀도(Predision), 재현율(Recall) 값을 알 수 있음

# ### 오차행렬 상에서 정확도
# 
# * 정확도(Accuracy) = 예측 결과와 실제 값이 동일한 건수 / 전체 데이터 수
# 
#     $ = \frac{TN + TP}{ TN + FP + FN + TP }$

# ### 불균형한 이진 분류 모델 
# 
# * 많은 데이터 중에서 중점적으로 찾아야 하는 매우 적은 수의 결과 값에 Positive를 설정해 1 값을 부여하고
# * 그렇지 않은 경우는 Negative로 0을 부여하는 경우가 많음  
# 
# 예1: 사기 행위 예측 모델
# * 사기 행위 : Positive 양성으로 1
# * 정상 행위 : Negative 음성으로 0  
#     
# 예2 : 암 검진 예측 모델
# * 양성 : Positive 양성으로 1
# * 음성 : Negative 음성으로 0 

# ### 불균형한 이진 분류 데이터 세트에서 정확도의 맹점
# 
# **Positive 데이터 건수가 매우 작아서 Positive 보다는 Negative로 예측 정확도가 높아지는 경향이 발생**  
# 
# - 10,000 건의 데이터 세트에서 9,900 건이 Negative이고 100건이 Positive라면 Negative로 예측하는 경향이 더 강해져서 TN은 매우 커지고 TP는 매우 작아지게 됨  
# 
# - 또한 Negative로 예측할 때 정확도가 높기 때문에 FN(Negative로 예측할 때 틀린 데이터 수)이 매우 작고, Positive로 예측하는 경우가 작기 때문에 FP 역시 매우 작아짐  
# 
# - 정확도 지표는 비대칭한 데이터 세트에서 Positive에 대한 예측 정확도를 판단하지 못한 채 Negative에 대한 예측 정확도만으로도 분류의 정확도가 매우 높게 나타나는 수치적인 판단 오류를 일으키게 됨  
# 
# 
# **불균형한 데이터 세트에서 정확도보다 더 선호되는 평가 지표**
# - 정밀도(Predision)와 재현율(Recall) 

# # 정밀도(Precision)와 재현율(Recall)

# ### 정밀도(Predision)와 재현율(Recall)
# * Positive 데이터 세트의 예측 성능에 좀 더 초점을 맞춘 평가 지표
# * 앞의 MyFakeClassifier는 Positive로 예측한 TP값이 하나도 없기 때문에
# * 정밀도와 재현율 값이 모두 0

# **정밀도와 재현율 계산 공식**
# * 정밀도 = TP / (FP + TP)
# * 재현율 = TP / (FN + TP)

# ![image-2.png](attachment:image-2.png)

# ### 정밀도 : TP / (FP + TP)
# * 예측을 Positive로 한 대상 중에 
# * 예측과 실제 값이 Positive로 일치한 데이터의 비율
# * 예측한 양성 대 예측한(맞춘) 양성
# * 공식의 분모인 (FP + TP)는 예측을 Positive로 한 모든 데이터 건수 (예측한 양성)
# * 분자인 TP는 예측과 실제 값이 Positive로 일치한 데이터 건수 (맞춘 양성)
# * Positive 예측 성능을 더욱 정밀하게 측정하기 위한 평가 지표로 
# * 양성 예측도라고도 불림

# ### 재현율 : TP / (FN + TP)
# * 실제값이 Positive인 대상 중에
# * 예측과 실제 값이 Positive로 일치한 데이터의 비율
# * 실제 양성 대 예측한(맞춘) 양성 비율
# * 공식의 분모인 (FN + TP)는 실제값이 Positive인 모든 데이터 건수 (실제 양성)
# * 분자인 TP는 예측과 실제 값이 Positive로 일치한 데이터 건수 (맞춘 양성)
# * 민감도(Sensitivity) 또는 TPR(True Positive Rate)이라고도 불림

# 보통은 재현율이 정밀도보다 상대적으로 중요한 업무가 많지만  
# 정밀도가 더 중요한 지표인 경우도 있음
# 
# 예: 스팸메일 여부를 판단하는 모델
# * 실제 Positive인 스팸메일을 Negative인 일반 메일로 분류하더라도
# * 사용자가 불편함을 느끼는 정도이지만
# * 실제 Negative인 일반 메일을 Positive인 스팸 메일로 분류할 경우
# * 메일을 아예 받지 못하게 되어 업무에 차질이 생길 수 있음

# **재현율이 상대적으로 더 중요한 지표인 경우**
# * 실제 Positive 양성인 데이터 예측을 Negative로 잘못 판단하게 되면 
# * 업무상 큰 영향이 발생하는 경우
# 
# **정밀도가 상대적으로 더 중요한 지표인 경우**
# * 실제 Negative 음성인 데이터 예측을 Positive 양성으로 잘못 판단하게 되면
# * 업무상 큰 영향이 발생하는 경우

# ### 재현율과 정밀도의 보완적 관계
# * 재현율과 정밀도 모두 TP를 높이는 데 동일하게 초점을 맞춤
# 
# 
# * 재현율은 FN(실제 Positive, 예측 Negative)를 낮추는데 초점을 맞추고
# * 정밀도는 FP를 낮추는데 초점을 맞춤
# 
# 
# * 재현율과 정밀도는 서로 보완적인 지표로 분류의 성능을 평가하는데 적용
# * 가장 좋은 성능 평가는 재현율과 정밀도 모두 높은 수치를 얻는 것
# * 반면에 둘 중 어느 한 평가 지표만 매우 높고, 다른 수치는 매우 낮은 결과를 나타내는 경우는 바람직하지 않음

# ### MyFakeClassifier의 예측 결과로 정밀도와 재현율 측정

# 타이타닉 예제로 오차 행렬 및 정밀도, 재현율 구해서 예측 성능 평가
# * 사이킷런 API 사용
#     - 정밀도 계산 : precision_score() 
#     - 재현율 계산 : recall_score()
#     - 오차행렬 : confusion_matrix()
# 
# 평가 간편 적용하기 위한 함수 작성
# * confusion_matrix / precision / recall 등의 평가를 한꺼번에 호출 
# 
# 타이타닉 데이터를 로지스틱 회귀로 분류 수행

# In[37]:


# 정밀도와 재현율 계산에 사용되는 예측값
# 앞에서 Dummy Classifier로 학습후 예측한 값 : fakepred
# (앞에 다 있는 내용인데 흩어져 있어서
# 정밀도와 재현율 계산을 위해 다시 모아서 적음)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
digits = load_digits()
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)
fake_clf = MyFakeClassifier()
fake_clf.fit(X_train, y_train)
fakepred = fake_clf.predict(X_test)


# In[38]:


# 참고 : fakepred 값 확인 
# fakepred # (모두 False)
fakepred.astype(int).sum()


# In[40]:


# 정밀도와 재현율 계산
# 정밀도 계산 : precision_score(실제값, 예측값)
# 재현율 계산 : recall_score(실제값, 예측값)
from sklearn.metrics import accuracy_score, precision_score, recall_score

print('정밀도 :', precision_score(y_test, fakepred))
print('재현율 :', recall_score(y_test, fakepred))


# ### 오차행렬, 정확도, 정밀도, 재현율을 한꺼번에 계산하는 함수

# In[50]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)  # 오차행렬
    accuracy = accuracy_score(y_test, pred)     # 정확도
    precision = precision_score(y_test, pred)    # 정밀도
    recall = recall_score(y_test, pred)         # 재현율
    
    print('오차행렬')
    print(confusion)
    print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f}'.format(accuracy, precision, recall))


# **앞의 타이타닉 데이터 세트 전처리 작업 내**

# In[43]:


# 타이타닉 데이터 세트 전처리 작업 내용 (앞에서 했음)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행. 
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


# In[51]:


# 로지스틱 회귀 기반으로
# 타이타닉 생존자 예측하고
# confusion matrix, accuracy, precision, recall 평가 수행

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

# 원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할. 
titanic_df = pd.read_csv('data/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df,                                                     test_size=0.20, random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test , pred)


# ## Precision/Recall Trade-off

# **정밀도 / 재현율 트레이드 오프(Trade-off)**
# * 업무에 따라 정밀도/재현율 중요도 다름
# * 분류하려는 업무 특성상 정밀도 또는 재현율이 특별히 강조돼야 할 경우
# * 분류의 결정 임계값(Threshold)을 조정해서 정밀도 또는 재현율의 수치를 높일 수 있음
# 
# * 정밀도와 재현율은 상호 보완적인 평가 지표이기 때문에 어느 한쪽을 강제로 높이면 다른 하나의 수치는 떨어지는데 이를 정밀도/재현율의 트레이드 오프라고 함

# ### predict_proba( ) 메소드

# 타이타닉 생존자 데이터를 학습한 LogisticRegression 객체에서  
# predict_proba() 메서드를 수행한 뒤 반환 값 확인하고  
# predict() 메서드와 결과 비교  
# 앞 예제에 이어서 코드 작성

# In[53]:


# lr_clf = LogisticRegression()
# predict_proba(테스트 피처 데이터 세트) : 예측 확률 반환

pred_proba = lr_clf.predict_proba(X_test)
pred_proba[:10]


# predict_proba() 결과 설명 : 예측 확률 array  
# 첫 번째 칼럼은 0 Negative의 확률  
# 두 번째 칼럼은 1 Positive의 확률  
# 반환 결과인 ndarray는 0과 1에 대한 확률을 나타내므로  
# 첫 번째 칼럼 값과 두 번재 칼럼 값을 더하면 1이 됨  
# [0.46162417 + 0.53837583] = 1

# In[56]:


# predict(테스트 피처 데이터 세트) : 예측 결과 클래스 값 반환
pred = lr_clf.predict(X_test)
pred


# In[58]:


# 예측 확률 array 와 예측 결과값 array 를 concatenate 하여 예측 확률과 결과값을 한눈에 확인
pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1, 1)], axis=1)

print('두개의 class 중에서 더 큰 확률을 클래스 값으로 예측')
print(pred_proba_result[:10])


# ### Binarizer 클래스 활용

# 사이킷런의 Binarizer 클래스 이용해서  
# 분류 결정 임계값을 조절하여  
# 정밀도와 재현율의 성능 수치를 상호 보완적으로 조정 가능

# Binarizer 클래스 이용 예측값 변환 예제
# * threshold 변수를 특정 값으로 설정하고
# * Binarizer 클래스의 fit_transform() 메서드를 이용해서
# * 넘파이 ndarray 입력 값을 지정된 threshold보다 같거나 작으면 0 값으로,
# * 크면 1값으로 변환해서 반환

# In[62]:


from sklearn.preprocessing import Binarizer

X = [[ 0.5, -1,  2],
     [ 2,  0,  0],
     [ 0,  1.1, 1.2]]

# threshold 기준값보다 같거나 작으면 0을, 크면 1을 반환
binarizer = Binarizer(threshold=1.0)                     
print(binarizer.fit_transform(X))


# **분류 결정 임계값 0.5 기반에서 Binarizer를 이용하여 예측값 변환**

# In[63]:


from sklearn.preprocessing import Binarizer

#Binarizer의 threshold 설정값. 분류 결정 임계값 = 0.5로 설정.  
c_threshold = 0.5

# predict_proba( ) 반환값([0확률 1확률])의 두번째 컬럼 , 
# 즉 Positive 클래스 컬럼 하나만 추출하여 Binarizer를 적용
pred_proba_1 = pred_proba[:,1].reshape(-1,1)

bina = Binarizer(threshold=c_threshold).fit(pred_proba_1) 
custom_predict = bina.transform(pred_proba_1)

get_clf_eval(y_test, custom_predict)

# 앞에서 predict()로 구한 결과와 동일


# **분류 결정 임계값을 0.4로 변경**

# In[64]:


custom_threshold = 0.4

# predict_proba( ) 반환값([0확률 1확률])의 두번째 컬럼 , 
# 즉 Positive 클래스 컬럼 하나만 추출하여 Binarizer를 적용
pred_proba_1 = pred_proba[:,1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) 
custom_predict = binarizer.transform(pred_proba_1)

get_clf_eval(y_test, custom_predict)

# 임계값을 낮추니까 정밀도는 떨어지고 재현율 값은 올라감


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# **여러개의 분류 결정 임곗값을 변경하면서  Binarizer를 이용하여 예측값 변환**

# In[65]:


# 테스트를 수행할 모든 임곗값을 리스트 객체로 저장. 
thresholds = [0.4, 0.45, 0.50, 0.55, 0.60]

def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값:',custom_threshold)
        get_clf_eval(y_test , custom_predict)

get_eval_by_threshold(y_test ,pred_proba[:,1].reshape(-1,1), thresholds )

# 정밀도 / 재현율 트레이드 오프 
# - 한 쪽을 향상시키면 다른 수치 감소하니까 적당한 수치 선택


# ### 임곗값에 따른 정밀도-재현율 값 추출
# - precision_recall_curve( ) 를 이용

# **precision_recall_curve( 실제값, 레이블 값이 1일 때의 예측 확률값)**
# - 정밀도, 재현율, 임계값을 ndarray로 반환
# - 임계값 : 일반적으로 0.11~0.95 범위
# - 정밀도와 재현율의 임계값에 따른 값 변화를 곡선 형태의 그래프로 시각화하는데 이용

# ### 예제
# - 반환되는 임계값이 너무 작은 값 단위로 많이 구성되어 있음
# - 반환된 임계값의 데이터 143건(교재 147건)인데
# - 임계값을 15단계로 해서 샘플로 10건만 추출
# - 좀 더 큰 값의 임계값과 그때의 정밀도와 재현율 확인

# In[70]:


from sklearn.metrics import precision_recall_curve

# 레이블 값이 1일때의 예측 확률을 추출 
pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1] 

# 실제값 데이터 셋과 레이블 값이 1일 때의 예측 확률을 precision_recall_curve 인자로 입력 
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1)
print('임계값 shape: ', thresholds.shape)
print('정밀도 shape: ', precisions.shape)
print('재현율 shape: ', recalls.shape)

idx = np.arange(0, thresholds.shape[0], 15)
print('sample index:', idx)
print('임계값 sample: ', np.round(thresholds[idx], 3))
print('정밀도 sample: ', np.round(precisions[idx], 3))
print('재현율 sample: ', np.round(recalls[idx], 3))


# In[50]:


#반환된 임계값 배열 행이 143건으로
# 임계값을 15단계로 해서 샘플로 10건만 추출
thr_index = np.arange(0, thresholds.shape[0], 15)
print('샘플 추출을 위한 임계값 배열의 index 10개:', thr_index)
print('샘플용 10개의 임곗값: ', np.round(thresholds[thr_index], 2))

# 15 step 단위로 추출된 임계값에 따른 정밀도와 재현율 값 
print('샘플 임계값별 정밀도: ', np.round(precisions[thr_index], 3))
print('샘플 임계값별 재현율: ', np.round(recalls[thr_index], 3))


# ### 임곗값의 변경에 따른 정밀도-재현율 변화 곡선

# In[51]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

# y_test : 실제 값  pred_proba_c1: 예측 확률 값
def precision_recall_curve_plot(y_test , pred_proba_c1): 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)

    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0] # (143,)에서 143 추출
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision') 
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')
 
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))

    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()

precision_recall_curve_plot( y_test, lr_clf.predict_proba(X_test)[:, 1] )


# 정밀도와 재현율 조합
# - Positive 예측의 임계값에 따라 정밀도와 재현율 수치가 변경
# - 임계값은 업무 환경에 맞게 정밀도와 재현율의 수치를 상호 보완할 수 있는 수준에서 적용되어야 함
# - 단순히 하나의 성능 지표 수치를 높이기 위한 수단으로 사용돼서는 안 됨
# 
# 분류의 종합적인 성능 평가에 사용하기 위해서는  
# 정밀도와 재현율의 수치를 적절하게 조합하는 것이 필요함

# # F1 Score

# ### F1 Score
# - 정밀도와 재현율의 조화평균
# - 정밀도와 재현율이 어느 한족으로 치우치지 않는 수치를 나타낼때 상대적으로 높은 값을 가짐

# ### F1 Score 공식
# ![image-2.png](attachment:image-2.png)

# ### 예 : 두 예측 모델 비교  
# A 예측 모델
# - 정밀도 : 0.9
# - 재현율 : 0.1 (극단적 차이)
# - F1 스코어 : 0.18
# 
# B 예측 모델
# - 정밀도 : 0.5
# - 재현율 : 0.5 (큰 차이 없음)
# - F1 스코어 : 0.5 
# 
# B모델의 FI 스코어가 A모델에 비해 매우 우수

# In[71]:


# 사이킷런의 F1 스코어 API : f1_score()
from sklearn.metrics import f1_score 

f1 = f1_score(y_test , pred)
print('F1 스코어: {0:.4f}'.format(f1))


# ### 타이타닉 생존자 예측에서 F1 스코어
# - 임계값을 변화시키면서 F1 스코어를 포함한 평가 지표 구하기
# - 임계값 0.4~0.6별로 정확도, 정밀도, 재현율, F1 스코어 확인

# In[73]:


def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    
    # F1 스코어 추가
    f1 = f1_score(y_test,pred)
    print('오차 행렬')
    print(confusion)
    # f1 score print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'
          .format(accuracy, precision, recall, f1))


# In[74]:


# 임계값 0.4~0.6별로 정확도, 정밀도, 재현율, F1 스코어 확인
thresholds = [0.4 , 0.45 , 0.50 , 0.55 , 0.60]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)


# # G measure

# - 정밀도와 재현율의 기하평균
# 
# - $ G = \sqrt{Precision × Recall}$

# # ROC Curve와 AUC

# ### ROC(Reciver Operating Characteristic)

# - 수신자 판단 곡선
# - 2차대전 때 통신장비 성능평가를 위해 고안된 척도
# - 의학분야에서 많이 사용
# - ML의 이진분류 모델의 예측 성능의 중요 평가지표
# 
# 
# - FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)가 어떻게 변하는지를 나타내는 곡선
#     - FPR이 X축, TPR이 Y축
#     
# 
# - TPR : 재현율과 같으며, 민감도(Sensitivity)라 부름
#     - 실제값 Positive(양성)가 정확히 예측되어야 하는 수준
#         - 질병이 있는 사람이 질병이 있는 것(양성)으로 판정 
#         
# - FPR : 1-특이성(Specificity)
#     - 질병이 없는 건강한 사람이 질병이 있는 것으로 예측되는 수준
#     - 특이성 : 실제값 Negative(음성)가 정확히 예측되어야 하는 수준
#         - 질병이 없는 건강한 사람은 질병이 없는 것(음성)으로 판정 

# ![image-3.png](attachment:image-3.png)

# - FPR은 0부터 1까지 변경하면서 TPR의 변화 값을 구함
#     - 분류 결정 임계값(Positive 예측값을 결정하는 기준)을 변경하면서 결정
#     
#     
# - FPR을 0으로 만들려면 분류 결정 임계값을 1로 지정
#     - Positive 예측 기준이 높아 데이터를 Positive로 예측할 수 없음
#     - FPR이 0인 경우 Positive를 예측할 수 없어 FPR이 0이 됨
#     
#     
# - FPR을 1로 만들려면 분류 결정 임계값을 0으로 지정하여 TN을 0으로 만들면 됨
#     - 분류기의 Positive 확률기준이 너무 낮아 다 Positive로 예측
#     - Negative를 예측할 수 없으므로 TN이 0이 되고 FPR은 1이 됨

# ![image.png](attachment:image.png)

# https://hsm-edu.tistory.com/1033

# ### AUC(Area Under the Curve)
# - ROC 곡선 아래 면적
# - 대각선의 직선에 대응되면 AUC는 0.5
# - 1에 가까울수록 좋은 수치
# - FPR이 작을 때 얼마나 큰 TPR을 얻는지에 따라 결정
# 
# ![image.png](attachment:image.png)

# In[75]:


from sklearn.metrics import roc_curve

pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1] 

fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)

# thresholds[0]은 max(예측확률)+1로 임의 설정되는데
# 이를 제외하기 위해 np.arange는 1부터 시작
thr_index = np.arange(1, thresholds.shape[0], 5)

print('샘플 추출을 위한 임곗값 배열의 index :', thr_index)
print('샘플용 임곗값: ', np.round(thresholds[thr_index], 2))
# 교재에서는 10개. 실제 11개

# 5 step 단위로 추출된 임계값에 따른 FPR, TPR 값
print('샘플 임곗값별 FPR: ', np.round(fprs[thr_index], 3))
print('샘플 임곗값별 TPR: ', np.round(tprs[thr_index], 3))


# In[61]:


def roc_curve_plot(y_test, pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음
    fprs, tprs, thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs, tprs, label='ROC')
    
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Specificity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()
    
roc_curve_plot(y_test, lr_clf.predict_proba(X_test)[:, 1] )


# In[57]:


# 타이타닉 생존자 예측 로지스틱 회귀 모델의 ROC AUC 값 확인

from sklearn.metrics import roc_auc_score

pred_proba = lr_clf.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_proba)
print('ROC AUC 값: {0:.4f}'.format(roc_score))


# In[58]:


# get_clf_eval() 변경 
# ROC-AUC 추가 : 예측 확률값을 기반으로 계산되므로
# 매개변수 pred_proba=None 추가
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test,pred)
    
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},        F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc)) 


# In[59]:


# 변경된 get_clf_eval() 호출 시 pred_proba_c1 추가
def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값:',custom_threshold)
        get_clf_eval(y_test , custom_predict, pred_proba_c1)


# In[60]:


# 임계값 0.4~0.6별로 정확도, 정밀도, 재현율, F1 스코어, ROC AUC 확인
thresholds = [0.4 , 0.45 , 0.50 , 0.55 , 0.60]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)


# In[ ]:




