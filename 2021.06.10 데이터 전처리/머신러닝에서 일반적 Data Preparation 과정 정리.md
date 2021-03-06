# 머신러닝에서 일반적 Data Preparation 과정 정리



1. 데이터 준비 과정의 중요성
2. 결측치의 처리 방법
3. 특징추출(Recursive Feature Elimination)
4. 데이터 정규화
5. 원 핫 인코딩으로 범주 변환
6. 숫자 변수의 범주형 변수로 변환
7. PCA를 통한 차원 축소



## 데이터 준비 과정

### 원시 데이터를 모델링에 적합한 형식으로 변화하는 작업으로 예측 모델 생성 프로젝트에서 가장 중요한 부분이며 가장 많은 시간이 소요



## 결측값 처리

### 많은 기계 학습 알고리즘이 결측값이 있는 데이터를 지원하지 않기에 누락된 값을 다루는 기술이 필요함. 누락된 값의 위치에 관련 데이터값(예 : 평균, 최빈값, 중앙값 등)을 넣어 데이터 대치를 진행함



### 특징 추출(Recursive Feature Elimination)

### 예측 모델을 개발할 때 입력 변수의 수를 줄이는 프로세스로

### 모델링의 계산 비용을 줄이고 경우에 따라 성능을 향상시킬 수 있음

#### RFE & PCA 두 가지 모델을 통해 가장 영향력 있는 특징(변수) 추출이 가능



#### RFE와 PCA의 차이점 

- RFE의 경우, 알고리즘 반환값에 컬럼(변수)를 확인할 수 있으나, PCA는 정확한 변수 확인이 어려움



