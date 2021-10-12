#!/usr/bin/env python
# coding: utf-8

# ## 사이킷런으로 수행하는 타이타닉 생존자 예측

# * 캐글에서 제공하는 타이타닉 탑승자 데이터 기반으로
# * 생존자 예측을 사이킷런으로 수행
# 
# * 타이타닉 생존자 데이터
#     - 머신러닝에 입문하는 데이터 분석가/과학자를 위한 기초 예제로 제공
#     - 많은 캐글 이용자가 자신의 방법으로 타이타닉 생존자 예측을 수행하고
#     - 수행 방법을 캐글에 공유
# 
# * 캐글 : 데이터 분석 오픈 포탈
#     - 세계적인 ML 기반 분석 대회를 온라인 상에서 주관
#     
# 캐글사이트 : https://www.kaggle.com/c/titanic/data
# ![image-2.png](attachment:image-2.png)

# ## 변수 별 정보

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

titanic_df = pd.read_csv('data/titanic_train.csv')
titanic_df.head(3)


# * Passengerid: 탑승자 데이터 일련번호
# * survived: 생존 여부, 0 = 사망, 1 = 생존
# * Pclass: 티켓의 선실 등급, 1 = 일등석, 2 = 이등석, 3 = 삼등석
# * sex: 탑승자 성별
# * name: 탑승자 이름
# * Age: 탑승자 나이
# * sibsp: 같이 탑승한 형제자매 또는 배우자 인원수
# * parch: 같이 탑승한 부모님 또는 어린이 인원수
# * ticket: 티켓 번호
# * fare: 요금
# * cabin: 선실 번호
# * embarked: 중간 정착 항구 C = Cherbourg, Q = Queenstown, S = Southampton

# In[2]:


titanic_df.shape


# In[3]:


# 데이터 칼럼 타입 확인
titanic_df.info()


# 데이터 정보 확인
# - RangeIndex: 891 entries, 0 to 890 : 전체 행 (891개 행)
# - Data columns (total 12 columns): 칼럼 수 (12개)
# - float64 : 2개
# - int64 : 5개
# - object(string) : 5개
#  - (판다스는 넘파이 기반으로 만들어졌는데
#  - 넘파이의 String 타입의 길이 제한이 있기 때문에 
#  - 이에 대한 구분을 위해 object 타입으로 명기)
# - Age : 714개 (Null값(NaN): 177개)
# - Cabin : 204개 (Null값(NaN): 687개)
# - Embarked : 889개 (Null값(NaN): 2개)

# ## 결측치 파악

# In[4]:


# Null 값 개수 확인
titanic_df['Age'].isnull().sum()


# In[5]:


titanic_df['Cabin'].isnull().sum()


# In[6]:


titanic_df['Embarked'].isnull().sum()


# ##  데이터 전처리 : 결측치 처리

# ### NULL 컬럼들에 대한 처리
# 
# - 사이킷 머신러닝 알고리즘은 Null 값을 허용하지 않으므로
# - Null 값을 어떻게 처리할지 결정
# - DataFrame()의 fillna() 메서드를 사용해서 
# - Null 값을 평균 또는 고정 값으로 변경
# - Age : 평균 나이로 변경
# - 나머지 칼럼 : 'N'으로 변경

# In[7]:


# Null 처리

titanic_df['Age'].fillna(titanic_df['Age'].mean, inplace=True)
titanic_df['Cabin'].fillna("N", inplace=True)
titanic_df['Embarked'].fillna("N", inplace=True)

# 모든 칼럼의 Null 값을 합산해서 Null 값이 없는지 확인
titanic_df.isnull().sum()


# ### 문자열 변수(피처) 빈도 분포 확인 : value_counts()

# In[8]:


# 문자열 피처 (Sex, Cabin, Embarked) 값 분류 확인
print('Sex분포: \n', titanic_df['Sex'].value_counts())
print('\nCabin분포: \n', titanic_df['Cabin'].value_counts())
print('\nEmbarked분포: \n', titanic_df['Embarked'].value_counts())


# ### 문자열 변수 Cabin값 변경

# In[9]:


# Cabin 칼럼 값 중에서 첫 번째 문자만 추출
titanic_df['Cabin'].str[:1]


# In[10]:


# Cabin 값을 선실등급만으로 표기 (선실 번호 제외)
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]

# 선실등급 별 개수 확인
titanic_df['Cabin'].value_counts()


# ## 성별에 따른 생존자수 분포(EDA)

# 머신러닝 알고리즘 적용해서 예측 수행 전에 데이터 탐색
# - 어떤 유형의 승객이 생존 확률이 높았는지 확인
# - 성별이 생존 확률에 어떤 영향을 미쳤는지 확인
# - 성별에 따른 생존자 수 비교

# In[11]:


# 성별(Sex) 분포 확인
titanic_df.groupby('Sex')['Sex'].count()

# 생존(Survived) 분포 확인
titanic_df.groupby('Survived')['Survived'].count()


# In[12]:


# 성별(Sex) 생존(Survived) 확인
# 사망 : 0
# 생존 : 1
# Survived 칼럼 : 레이블로 결정 클래스 값

titanic_df.groupby(['Sex','Survived'])['Survived'].count()


# In[13]:


# 성별 생존자 : 막대 그래프 (barplot)
sns.barplot(x='Sex', y = 'Survived', data=titanic_df)


# In[14]:


# 객실 등급별/성별 생존 확률
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)

