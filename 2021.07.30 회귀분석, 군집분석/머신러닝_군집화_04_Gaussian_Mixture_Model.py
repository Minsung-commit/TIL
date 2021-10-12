#!/usr/bin/env python
# coding: utf-8

# # GMM(Gaussian Mixture Model)

# ## GMM(Gaussian Mixture Model) 소개
# 
# GMM 군집화
# - 군집화를 적용하고자 하는 데이터가 여러 개의 가우지안 분포를 가진 데이터 집합들이 섞여서 생성된 것이라는 가정하에 군집화를 수행하는 방식
# 
# 
# - 가우지안(Gaussian) 분포 : 정규분포
#     - 좌우대칭 종모양
#     - 연속형 확률 분포
#     - 평균과 표준편차에 따라 분포의 모양 결정
#     - 표준정규분포 : 평균이 0, 표준편차가 1인 정규분포
# ![image.png](attachment:image.png)
#     
# 
# - GMM은 데이터를 여러 개의 가우지안 분포가 섞인 것으로 간주하고, 섞인 데이터 분포에서 개별 유형의 가우지안 분포를 추출함

# ### 예. 세 개의 가우지안 분포 A,B,C를 가진 데이터 세트가 있다고 가정
# 
# ![image.png](attachment:image.png)

# - GMM는 아래 왼쪽 그림과 같은 데이터 세트에서 오른쪽 그림과 같이 여러 개의 정규분포 곡선을 추출하고 개별 데이터가 이 중 어떤 정규 분포에 속하는지 결정하는 방식
# ![image-5.png](attachment:image-5.png)

# ### GMM의 모수 추정
# 
# - 개별 정규분포의 평균과 분산
# - 각 데이터가 어떤 정규 분포에 해당되는지의 확률
#     
# ### GMM의 모수 추정 방법
# - EM(Expectation and Maximization) 방법을 적용하여 모수 추정을 함
# 
# - 사이킷런은 GMM의  EM 방식을 통한 모수 추정을 위해 **`GaussianMixture`** 클래스 지원

# ### GMM은 확률기반 군집화이고, K-평균은 거리기반 군집화

# ### GaussianMixture 클래스
# 
# - `sklearn.mixture` 패키지
# 
# 
# - 주요 파라미터
#     - `n_components` : Gaussian Mixture 모델의 총 갯수
#     
# 
# - `fit(피처데이터 세트), predict(피처데이터 세트)` 메서드로 군집을 결정
# 
# 

# ## GMM 을 이용한 붓꽃 데이터 셋 클러스터링

# ### 붓꽃 데이터 로드

# In[1]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()
feature_names = ['sepal_length','sepal_width','petal_length','petal_width']

# 보다 편리한 데이타 Handling을 위해 DataFrame으로 변환
irisDF = pd.DataFrame(data=iris.data, columns=feature_names)
irisDF['target'] = iris.target


# ### 붓꽃 데이터의 GMM 군집화
# - n_components는 3으로 지정
# - 군집화 결과는 irisDF의 'gmm_cluster' 필드로 추가 저장
# - GMM 군집 결과와 target 값과 비교

# In[2]:


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

# 클러스터링 결과를 irisDF 의 'gmm_cluster' 컬럼명으로 저장
irisDF['gmm_cluster'] = gmm_cluster_labels
irisDF['target'] = iris.target

# target 값에 따라서 gmm_cluster 값이 어떻게 매핑되었는지 확인. 
iris_result = irisDF.groupby(['target'])['gmm_cluster'].value_counts()
print(iris_result)


# - target 1만 cluster 2로 45개(90%), 1로 5개(10%) 매핑
# - target 0는 cluster 0으로, target 2는 cluster 1로 모두 매핑

# ### K-평균 군집화 수행

# In[3]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init='k-means++', 
                max_iter=300,random_state=0).fit(iris.data)
kmeans_cluster_labels = kmeans.predict(iris.data)
irisDF['kmeans_cluster'] = kmeans_cluster_labels
iris_result = irisDF.groupby(['target'])['kmeans_cluster'].value_counts()
print(iris_result)


# - 붓꽃 데이터는 GMM 군집화가 K-평균 군집화보다 더 효과적임
# 
# - K-평균은 평균 거리 중심으로 중심을 이동하면서 군집화를 수행하는 방식이므로 개별 군집 내의 데이터가 원형으로 흩어져 있는 경우에 매우 효과적으로 군집화가 수행될 수 있음

# ## GMM와 K-평균의 비교

# - K-평균은 원형의 범위에서 군집화를 수행함
# - 데이터 세트가 원형의 범위를 가질수록 KMeans 군집화 효율이 더욱 높아짐

# ### 예. make_blobs()로  cluster_std=0.5인 군집 3개를 생성

# In[7]:


### 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 
### 인자로 받아 클러스터링 결과를 시각화하는 함수  

def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise=True
        else :
            cluster_legend = 'Cluster '+str(label)
        
        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70,                    edgecolor='k', marker=markers[label], label=cluster_legend)
        
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',                        edgecolor='k', marker='$%d$' % label)
    if isNoise:
        legend_loc='upper center'
    else: legend_loc='upper right'
    
    plt.legend(loc=legend_loc)
    plt.show()


# **길게 늘어난 타원형의 데이터 셋을 생성**

# In[6]:


from sklearn.datasets import make_blobs

# make_blobs() 로 300개의 데이터 셋, 3개의 cluster 셋, cluster_std=0.5 을 만듬. 
X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=0)

# 길게 늘어난 타원형의 데이터 셋을 생성하기 위해 변환함. 
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
# feature 데이터 셋과 make_blobs( ) 의 y 결과 값을 DataFrame으로 저장
clusterDF = pd.DataFrame(data=X_aniso, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y
# 생성된 데이터 셋을 target 별로 다른 marker 로 표시하여 시각화 함. 
visualize_cluster_plot(None, clusterDF, 'target', iscenter=False)


# ### KMeans를 적용하여 타원형 데이터셋에 대한 군집화 결과

# In[6]:


# 3개의 Cluster 기반 Kmeans 를 X_aniso 데이터 셋에 적용 
kmeans = KMeans(3, random_state=0)
kmeans_label = kmeans.fit_predict(X_aniso)
clusterDF['kmeans_label'] = kmeans_label

visualize_cluster_plot(kmeans, clusterDF, 'kmeans_label',iscenter=True)


#      => K-평균 결과 cluster 0과 1인 잘못 그룹화 됨

# ### GMM을 적용한 군집화

# In[8]:


# 3개의 n_components기반 GMM을 X_aniso 데이터 셋에 적용 
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_label = gmm.fit(X_aniso).predict(X_aniso)
clusterDF['gmm_label'] = gmm_label

# GaussianMixture는 cluster_centers_ 속성이 없으므로 iscenter를 False로 설정. 
visualize_cluster_plot(gmm, clusterDF, 'gmm_label',iscenter=False)


# ### K-평균과 GMM 군집화 결과 비교

# In[8]:


print('### KMeans Clustering ###')
print(clusterDF.groupby('target')['kmeans_label'].value_counts())
print('\n### Gaussian Mixture Clustering ###')
print(clusterDF.groupby('target')['gmm_label'].value_counts())


# - 긴 타원형 데이터셋에는 K-평균보다 GMM이 군집화 정확도가 더 높다
# - GMM은 K-평균보다 더 다양한 데이터 세트에 잘 적용될 수 있는 장점이 있으나,
# - 군집화를 위한 수행시간이 오래걸리는 단점이 있음
