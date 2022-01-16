![header](https://capsule-render.vercel.app/api?type=slice&color=auto&height=300&section=header&text=Today I Learned &fontSize=90)








# TIL(Today I Learned)

이민성 

데이터 분석 학습 기록 시작

2021.06.04~

# 최종 프로젝트
https://web.archive.org/web/20211010064815/http://3.35.94.153/homepage/
<img width="629" alt="배포" src="https://user-images.githubusercontent.com/85140314/136686504-e65e6668-c7df-4f6a-883a-b0f1128313fa.png">

<img width="995" alt="배포2" src="https://user-images.githubusercontent.com/85140314/136686525-af1d5a67-ed92-4bf9-9eaf-e4be5792ad6b.png">


## 2021.06.04

### 학습방법:  멀티캠퍼스 강의

### 학습내용

- TIL 개설 및 내용 업로드 방법 학습
- Git의 포트폴리오 활용과 중요성 학습
- 파이참 깃 터미널 연결 및 사용 방법 학습



#### 자기평가

##### Git의 중요성을 파악하였기에 TIL 활용법 연구 및 꾸준한 이용이 필요함



## 2021.06.05

### 학습방법: 유튜브 동영상강의 

### 학습내용

- 알고리즘의 기초 학습
  - 알고리즘의 기본 단어
    - 트리(루트노드, 단말노드, 크기, 깊이 등)
    - 이진 탐색 트리: 이진 탐색이 동작할 수 있도록 고안된 자료구조의 일종
      - 이진탐색이란? 정렬된 배열 중에서 탐색 범위를 좁혀가며 원하는 값을 찾는 탐색
    - 트리의 순회: 트리 자료구조에 포함된 노드를 특정한 방법으로 한 번 씩 방문하는 방법
    - 정렬 알고리즘(선택 정렬, 삽입 정렬, 퀵 정렬, 계수 정렬)



#### 학습평가

##### 알고리즘의 이해가 아직 완벽하게 되지 않았기에 추가 반복적인 학습이 필요할 것으로 생각되며, 관련하여 쉬운 예제를 풀어보는 학습을 진행해야 함.



## 2021.06.06

### 학습방법: 멀티캠퍼스 온라인 강의/ 유튜브

### 학습내용: 

- 시간복잡도 및 공간복잡도의 정의
- 선형탐색의 의미 파악
- 기본 자료구조인 스택과 큐의 의미
- 스택, 큐의 파이썬 구현 코드
- 우선순위 큐의 정의와 구현법(리스트 구현, 힙 정렬)



#### 학습평가

##### 우선순위 큐의 내용이 매우 난해함, 특히 힙 정렬의 내용은 거의 이해하지 못하였기에 힙 정렬과 관하여 보충 학습이 필요함



## 2021.06.07

### 학습방법: 멀티캠퍼스 강의 / 블로그

### 학습내용: 판다스(pandas) 기본 활용법

- reading and writing
  - pd.read(), pd.to_csv()
  - .head(), .tail(), .describe(), .info()
- Indexing, Selecting, and Assigning
  - .loc[], .iloc[]
  - dataframe[dataframe.col == "col"]
- Summary_mapping, applying
  - median(), mean(), unique(), value_counts()
  - centeting transformation = "value" - "value".mean()
  - idxmax(column, index(row)): it is used to return a certain index number which the maximum value has in a certain range(column, row)
  - map() for Series, apply() for DataFrame



#### 학습평가

##### apply와 map 메소드의 사용이 미숙함. 특히 두 함수를 적절히 응용하는 능력이 떨어지기에 다양한 예제를 확인할 필요가 있음.



## 2021.06.08

### 학습방법 : 멀티캠퍼스 강의 / 블로그

### 학습내용 : 넘파이 기반 이미지 크롭, 머신러닝 기초/데이터분석 패키지

- EDA 기초 학습(자료기반 인사이트 도출, 로우 스케일 등)
- 동영상 detecting 패키지 기초 실습

- Numpy 기반 이미지 크롭
  - opencv 
    - import cv2, cv2.imread(파일경로+파일명)
    - 변수a [00:00, 00:00] (세로, 가로)
    - cv2.imshow(새이름, 파일명)
    - cv2.imwrite(파일경로+파일명)
    - cv2.waitkey(0) #0 forever
    - cv2.destroyAllWindows()



#### 학습평가

##### 기초 통계지식 및 용어학습의 부족으로 인해 수업에 따라가는 것이 힘들었음. 내가 뭘하고 있었는지 잘 와닿지 않았음. 이에 따라 기초 통계 및 데이터 분석 학습을 위해 데이터 분석 기본서 및 기초 통계학 서적을 대여함.



## 2021.06.09

### 학습방법: 멀티캠퍼스 강의 / [이토록 쉬운 통계&R]

### 학습내용:

- 공분산/다중공선성/상관계수
- 머신러닝 툴 사용 연습
- 자유도, 머신러닝 분류모형 성능 지표 학습



#### 학습평가

##### pca툴 사용이 어려움, 기초 통계학 공부가 필요함 

##### 이를 위해 통계학 스터디 진행 매주 목요일 저녁 8시 진행



## 2021.06.10

### 학습방법: 멀티캠퍼스 강의

### 학습내용: 데이터 전처리

- 특징 추출(pca)과 특징 선택(rfe) 실습
- 결측값 제거(SimpleImputer)
- 차원축소의 개념과 머신러닝에서의 일반적인  Data Preparation과정 학습

#### 학습평가

##### 특징 추출과 특징 선택에 대한 개념이 확실히 잡혀있지 않기에 이 부분에 대한 추가학습이 필요함

## 2021.06.11

### 학습방법 : 멀티캠퍼스 강의

### 학습내용 : 범주 및 변수 변환

- 원 핫 인코딩 : 범주형 데이터 자료를 이진수로 변환하여 고유값을 부여함
- 라벨 인코딩 : 범주형 데이터 자료를 정수로 변환함
- KBinsDiscretizer : 연속형 데이터 자료를 숫자 범주형 자료로 변환함



#### 학습평가

##### 원 핫 인코딩과 라벨 인코딩을 선택하는 기준에 대해 확실한 기준이 애매모호하여 추가학습이 필요할 것으로 사료됨. 더불어 인코딩 실습을 추가적으로 진행하여 숙련도를 높여야 함.



## 2021.06.14

### 학습방법 : 멀티캠퍼스 강의

### 학습내용 : 데이터베이스, 관계형 데이터 베이스, SQL

- 데이터베이스 
- 관계형 데이터베이스
- SQL
- mySQL 설치



#### 학습평가

##### DB 학습 시작으로 mySQL 설치 및 기초 문법에 대해 학습함, Access와 Excel에서 SQL 문법을 접해 본 적이 있기 때문에 큰 무리없이 수업에 따라 갈 수 있었음.



## 2021.06.15

### 학습방법 : 멀티캠퍼스 강의

### 학습내용 : SQL 네트워크 연결, SQL 판다스 비교, JOIN 구문

- SQL 기본 설정 및 네트워크 연결 방법 학습 (쥬피터 노트북에서 mySQL db 연결하기)
- JOIN 구문 학습 
  - left join - select * from * left join *  on * where *
  - inner join - select * from * inner join * on * where *
- pandas에서 SQL과 같은 동작하기 - assign, agg, value_counts(), count()



#### 학습평가

##### FT님의 말씀 - 이해도를 높이기 위해선 다양한 시도, 구문을 사용해봐야 함. 프로젝트를 위해서도 끊임없는 연습이 필요함



## 2021.06.16

### 학습방법 : 멀티캠퍼스 강의

### 학습내용 : 데이터 시각화 matplotlib, seaborn

- matplotlib을 통한 데이터 시각화 실습
- seaborn을 활용한 데이터 시각화 실습
  - histplot, displot
  - boxplot
  - regplot
  - scatterplot



## 2021.06.17

### 학습방법 : 멀티캠퍼스 강의 / 통계학 스터디

### 학습내용 : 데이터 전처리 과정 복습 / PCA 심층 연구

- 데이터 전처리 과정 복습
  1. DB 데이터 셋 불러와 데이터 프레임 만들기
  2. 데이터프레임 결측치 확인
  3. object type 제거를 위해 replace 함수를 통한 바이너리, 라벨 인코딩 진행
  4. 인코딩된 데이터프레임의 결측치 제거  > Simple Imputer
  5. 정제된 데이터프레임에 RFE/Regression Feature Selection 적용(타겟 변수 설정 필요)
  6. 정제된 데이터 프레임에 PCA 적용(타겟 변수 설정 불필요)
  7. Nomalization/Standardization 적용하여 값 확인 
- PCA 심층 연구
  - PCA를 통해 만들어진 새로운 축은 기존의 축과 다른 축(기존 축의 이름을 사용할 수 없음)
  - PCA를 실제적으로 적용하는 모델링 학습이 필요함



#### 학습평가

##### 복습 내용에는 문제가 없었으나, PCA를 적극적으로 활용하기 위해선 심도 깊은 이해가 필요하다고 느껴짐 아이겐 벨류, 아이겐 벡터 등..



## 2021.06.18

### 학습방법 : 멀티캠퍼스 강의

### 학습내용 : PCA 활용 연구

1.  PCA 적용시 Standardization 적용 후 PCA 진행
2.  이를 바탕으로 하여 데이터 전처리 과정 재복습



#### 학습 평가

##### 0617(목) 수업과 동일한 내용으로 재진행 된 수업, PCA관련하여 많은 논의와 연구가 있었고 그에 따라 실제 데이터 전처리 작업시 PCA를 적용하는 방법에 대해 고민할 수 있었음.



## 2021.06.20(일)

### 학습방법 : 온라인 강의

### 학습내용 : 빅데이터 분석을 위한 통계 상식

- 통계 상식 1-6차시 까지의 내용
- 평균의 종류와 개념, 대표값 등
- 퍼센트의 정의와 퍼센트 해석시 주의사항
- 시각화, 그래프의 개념과 그래프 작성 및 해석시 주의사항
- 확률의 개념과 종류



#### 학습평가:

##### 기초 통계에 대한 내용으로 기본적으로 다 숙지하고 있는 내용이었으나 탄탄한 기본을 위해 한번 더 복습을 진행하였음.



## 2021.06.21(월)

### 학습방법 : 멀티캠퍼스 강의, 통계학 스터디 

### 학습내용 : HTML / Java script / 데이터 샘플링 /오버샘플링&언더샘플링

### 학습툴: Visual Studio, w3schools, ml5js, teachablemachine



- Visual Studio 기본 사용법 및 function 숙지
- HTML 기초 작성법 학습 및 실습
- Image classfication 실습
- Image classficatio(video) 실습
- Teachablemachine을 활용한 머신러닝 실습
- 데이터 샘플링 종류와 각 장단점 연구
- 머신러닝 내 Oversampling and Under sampling 사용법 및 기초 정의 학습



#### 학습평가:

머신러닝에 대한 기초 개념이 부족하여 스터디에 따라가는 것이 힘들었음. 머신러닝에 대한 기초 개념학습이 우선되어야 할 것 같다.



## 2021.06.22(화)

### 학습방법 : 멀티캠퍼스 강의

### 학습내용 : django를 활용한 web 프로그래밍 및  db 연결

### 학습툴 : django / visual studio

- anaconda 내에 django environment 설치
- visual studio를 통한 django 웹 프로그래밍
- django db  연결 실습
- django admin and staff 계정 생성



#### 학습평가 : 

django 활용 방식이 아직 정확하게 이해되지 않음.  



## 2021.06.23(수)

### 학습방법 : 멀티캠퍼스 강의 

### 학습내용 : django를 활용한 migration 및 api 사용법 학습

### 학습툴 : django / visual studio / mySQL workbench



- django migration 생성 및 migratting 실습
- api 를 통한 기초 웹 프로그래밍 실습



#### 학습평가:

##### 웹 프로그래밍 파트는 상당히 어려워 따라가기가 힘들었다. 사실 아직도 이해되지 않은 부분들이 많아 만약 엔지니어링 쪽으로 가게 된다면 이 부분에 대한 추가학습이 반드시 필요할 것이다.



## 2021.06.24(목)

### 학습내용 : 시계열 분석 / fbpromphet / stationarity

### 학습방법: 멀티캠퍼스 강의

### 학습툴 : 쥬피터 노트북 / python



- 실제 데이터 분석 예시를 통한 단/다변량 분석방법 확인
- fbpromphet 사용 코드 분석을 통한 fbpromphet의 특/장점 학습
- stationarity/noise 제거를 위한 방법론 학습



#### 학습평가: 

##### 실습 없이 예시를 보며 코드 분석을 진행하였다. 진정한 데이터 분석에 한걸음 더 다가갈 수 있었던 좋은 시간이었다.



## 2021.06.25(금)

### 학습내용 :  데이터분석 우수사례 분석 / zero-shot classification / Probabilistic_Layers_Regression

### 학습방법 : 멀티캠퍼스 강의

### 학습툴 : 쥬피터 노트북

- 사례분석을 통한 분석과정 및 인사이트 도출 방법 학습
- 학습데이터 없이도 분류를 할 수 있는 zero-shot classification 모델 실습
- probabilistic layers regression 모델 실습
- 시간과 사건에 따른 가능성 정의 학습(unknown/unknows - known-knows)



#### 학습평가:

##### probabilistic layers regression 모델에 대한 이해가 되지 않았기에 복습을 진행할 예정임, 추가적으로 본격적인 team PJT가 시작되어 그와 관련하여 pjt 계획을 수립하였음.



## 2021.06.26(토)

### 활동내용 : PJT 준비 /알고리즘

### 활동방법 : 멀티캠퍼스 강의 / 블로그 및 유튜브

### 활동툴 : 노션, 구글 드라이브, 깃허브

- PJT준비  - 프로젝트 1차 완료일을 13일로 설정하고 그에 맞춘 사전준비를 진행함(2021.06.26 PJT준비 폴더 참고)
- 멀티캠퍼스를 통해 알고리즘 학습 진행
  - 그래프 탐색_DFS&BFS
  - 다익스트라 알고리즘
  - 플로이드 워셜 알고리즘



#### 활동평가:

##### pjt 준비를 진행하였고 일요일 미팅을 통해 task list를 작성할 예정임, 알고리즘에 대한 기초지식이 탄탄하지 않음을 느낌



## 2021.06.28(월)

### 활동내용 : 빅데이터를 위한 통계 상식 학습 / PJT 회의

### 활동방법 : 멀티캠퍼스 강의

### 활동툴 : X



- 데이터 샘플링의 기초 개념과 전략 학습
- 상관관계에 대해 정의 학습
- 회의 내용 -  주제 구체화 및 분석 전략 수립



#### 활동 평가:

##### 프로젝트 주제에 대한 의견이 조율이 되지 않아, 해당 부분을 조율하는 것을 중점으로 함 그에 따라 최우선 과제를 설정하고 각자에게 업무 분장을 진행하였음.



## 2021.06.29(화)

### 활동내용 : 웹 이미지 크롤링 & pdf크롤링 

### 활동방법 : 멀티캠퍼스 강의

### 활동툴 : camelot / googleimagescrapper / selenium / microsoft power automate



- camelot을 활용한 pdf table 크롤링 실습
- google image scrapper 를 활용한 이미지 크롤링 실습
- selenium을 활용한 이미지 크롤링 실습
- power automate 사용법 및 크롤링 활용 실습



#### 활동평가 

##### 이미지 크롤링에서 더 나아가 power automate 툴을 다양하게 활용할 수 있는 능력을 기른다면, 다양한 영역에서 범용적으로 활용할 수 있다는 생각이 들었다. 특히나 automate를 활용하여 텍스트 크롤링을 하여 데이터 베이스를 구축하는 일을 한번 시도해보고 싶어졌다.



## 2021.06.30(수)

### 활동내용 : text crawling / logging 실습 / power automate 연습

### 활동방법 : 멀티캠퍼스 강의 / flow 제작 연습

### 활동툴 : Replit / power automate / eliot / logging / mySQL / lxml /feedparser



- eliot / logging을 이용한 log 데이터 추출 및 저장 실습
- Replit /  grep, sed 함수를 이용하여 텍스트 추출 실습 
- lxml.html / feedparser를 활용한 웹페이지 텍스트 추출 실습
- 추출된 텍스트 db 저장 방법 실습
- power automate flow 제작 연습



#### 활동평가

##### eliot/lxml/feedarser 등 다양한 방법을 배웠으나, splunk를 활용하면 빠르고 직관적으로 텍스트 크롤링이 가능하다고 함, 알아두면 좋으나 실제로 활용지수는 낮다고 한다. power automate를 혼자서 실습해봤으나 생각보다 많은 기능들이 탑재되어 있어 이 기능들을 숙지하는데에는 꽤 오래 걸릴 것 같다. 하지만 툴을 사용하는 재미는 여태까지 배웠던 툴 중 가장 좋다.



## 2021.07.01(목)

### 활동내용 :  정규표현식 / 크롤링 데이터 db 저장 심화 내용 학습 / 프로젝트 회의 진행

### 활동방법 : 멀티캠퍼스 강의

### 활동툴 :  쥬피터 노트북



- 정규표현식 학습 
- 크롤링 데이터 db 저장 심화 내용 학습 및 실습
- 폐업요인 리스트 작성을 주제로 프로젝트 미팅을 진행하였고, 해당 내용을 노션에 업로드하였음

#### 활동평가

##### 정규 표현식은 내용이 쉬워 금방 이해 했으나, 크롤링 데이터 db저장의 경우 100% 이해하진 못하였다. 강사님께서 말씀하시길 굉장히 원초적인 방법으로 접근하고 있기에 알아만 두고, 실제 현장에서는 휠씬 간단한 방법으로 진행한다고 하셨음.



## 2021.07.02(금)

### 활동내용 : Splunk 설치 및 기초 개념 학습 / PJT 진행 과정 발표

### 활동방법 : 멀티캠퍼스

### 활동툴 : Splunk



- 대용량 데이터의 처리, 시각화 등을 진행할 수 있는 Splunk의 설치와 기초 사용법 학습 진행
- Splunk tutorial data를 통한 데이터 활용, 시각화 실습 진행
- PJT 관련 회의의 진행을 맡아 변수 선정에 대해 논의하였고, 이제까지의 진행 내용을 발표하였음



#### 활동평가 :

##### Splunk에 대해 아직 학습량이 부족함을 느낌, 이해가 되지 않은 부분이 더 많았다. PJT 관련한 진행이 매우 더디다. 뚜렷한 주제 없이 아직 멤도는 느낌이 강하여, 이 점에 대해 강사님으로부터 타겟팅을 해야 한다는 피드백을 얻을 수 있었음.



## 2021-07-05(월)

### 활동내용 : 알고리즘 특강 / PJT 회의

### 활동방법 : 구글 미팅 / 멀티캠퍼스 강의

### 활동툴 : 파이썬/파이참



- 알고리즘 특강 진행
  - 자료구조와 알고리즘 개념 학습(리스트, 스택, 큐)
  - 선형리스트의  개념과 예를 익히고,  선형리스트의 생성과 삭제, 삽입 방법을 실습하였다.
  - 연결리스트의 개념과 예를 익히고, 연결리스트의 생성과 삭제, 삽입 방법에 대해 실습 진행
  - 스택의 개념과 생성방법 학습, 삽입, 삭제 실습 진행
- PJT 회의 진행
  - 주제 확정
  - 경제활동인구총조사 데이터셋 전처리 시작
  - 개별 EDA 진행 예정



#### 활동평가:

##### pjt 관련하여 소통이 제대로 이루어 지지 않아, 각 팀원이 생각하고 있던 조사&분석의 범위가 달라졌었다. 팀 프로젝트를 진행하는데에 있어 소통을 더 정확하고 꾸준히 할 수 있도록 하기 위해, 노션(notion)에 회의 내용을 올릴 때 구체적이고 자세하게 내용을 공유할 수 있도록 노력해야 할 것이다.



## 2021-07-06(화)

### 활동내용 : 알고리즘 특강 / 태블로 학습

### 활동방법 : 멀티캠퍼스 강의 / 유튜브 독학

### 활동툴 : 파이썬/파이참/태블로



- 태블로 설치 및 연결 방법 학습
- 태블로를 통한 데이터 시각화 및 애니메이션 표현법을 학습함
- 알고리즘 정렬 코드 실습
- 알고리즘 예제 문제 풀이



## 2021-07-07(수)

### 활동내용 : Nosql / splunk / knn / k-means

### 활동방법 : 멀티캠퍼스 강의

### 활동툴 : splunk / visualcode / 태블로



- splunk 재학습 
- Nosql 이론 및 db 역사 학습
- pjt 진행을 위한 knn / k-means 학습

#### 활동평가:

##### knn/k-means를 통한 결측치 제거를 진행해야 하기 때문에, 해당 내용에 대한 철저한 학습이 필요함



## 2021-07-08(목)

### 활동내용 : knn / mice 기법

### 활동방법 : 멀티캠퍼스 강의 / 독학

### 활동툴 : fancyimputer / visualcode / 태블로



- fancyimputer 내 내장 모듈인 IrativeImputer 와 KNN()을 활용하여 결측치 처리를 시도
- 각각의 사용조건에 맞추어 원핫 인코딩을 시도하였음
- 원핫 인코딩 후 결측치 값 대체 시도



#### 활동평가:

##### 결론부터 말하자면, 결국 모든 시도가 실패하였다. 실패 원인은 knn 및 mice 기법의 특징에 대한 이해부족이 큰 탓이었다. 특히 knn의 경우 categorical feature보단 numerical 에 더 적합하며, 원핫인코딩을 한다고 해서 곧바로 값을 넣을 수 있는 것이 아니라, 인코딩 된 값에서 결측치를 다시금 재입력해야하는 수고를 해야만 했다. 더 큰 문제는 knn은 비교,사용해야 하는 값이 많아질수록 복잡도가 높아져 많은 메모리를 소비하기때문에 컴퓨터에 큰 부담을 줄 수 있다는 사실이었다. 실제로 내 컴퓨터에선 메모리 부족으로 정상적으로 값이 인출되는지 조차 확인하지 못하였다. categorical feature에 있어 더 나은 결측치 처리 방법을 연구해보아야 할 것이다.



## 2021-07-11(일)

### 활동내용 : 태블로를 활용한 데이터 시각화

### 활동방법 : 태블로, 태블로 웹사이트 

### 활동툴 : 파이썬, 태블로



- PJT 를 위해 준비한 데이터셋 전처리 완료 후 해당 데이터셋을 통해 시각화 진행



#### 활동평가:

##### 태블로를 활용한 시각화 방법이 정말 다양하고 무궁무진하다는 것을 알게 되었다. 특히나 계산필드를 활용하여, 새로운 필드 생성을 할 수 있는 기능은 정말 사용하기 편리할 것으로 느껴졌고, GUI를 통한 데이터 시각화라서 그 활용도와 신속성이 타 도구들에 비해 내게 더 적합한 느낌이 든다.



## 2021-07-12(월)



- 연구모형 설정



## 2021-07-13(화)

### 활동내용 : PJT 진행



- 총생활인구 증감률을 상권별로 분석 진행
- 분석 결과를 태블로를 활용하여 시각화함



## 2021-07-14(수)

### 활동내용 : PJT - 사회적거리두기,공휴일 더미 생성 / 애자일프로젝트

### 활동방법 : 멀티캠퍼스 강의 / 블로그 활용

### 활동툴 : 파이썬, 쥬피터 랩



- 판다스 date_range() 함수를 활용한 일정 주기의 날짜 데이터 프레임 생성
- 생성된 날짜 프레임에 사회적 거리두기, 공휴일 범주형 변수 합치기
- 범주형 변수를 활용하기 위한 원 핫 인코딩 실행
- 애자일 프로젝트
  - 프로젝트 개발 방식의 기본 개념 및 종류 학습
- 노션 업데이트 
  - pjt 진행상황 및 발표일정 공유



#### 활동평가:

##### pjt 진행과 관련하여서 진행하면서도 모호한 부분이 많다. 이 부분을 해결하고 싶지만 지금은 도저히 방법을 모르겠다. 계속해서 관련 선행연구를 찾아봐야지...



## 2021-07-15(목)

### 활동내용 : PJT - 분기변수 생성, 분기별 사회적 거리두기 비율 설정, 매출액증감률 시각화

### 활동툴 : 태블로, 파이썬



- date_range()를 통해 분기변수 생성
- 사회적 거리두기 값 입력, 인코딩 진행
- 매출액 증감률 데이터 태블로 시각화



#### 활동평가:

##### PJT완료 기간이 얼마 남지 않았기 때문에 조금 더 타이트하게 해야 할 것 같다.



## 2021-07-16(금)

### 활동내용 : PJT - 더미변수 생성, 분기별 사회적 거리두기 비율 설정, 매출액증감률 시각화

### 활동툴 : 태블로, 파이썬



- 재난지원금, 할로윈,크리스마스, 사회적 거리두기 더미 생성
- 더미와 분기변수 join 진행



## 2021-08-22(일)

### 활동내용 : ADsP 시험 대비 공부



- 데이터 이론, 통계분석 파트 학습



## 2021-08-25 ~ 2021-10-08 

### 프로젝트 진행



- 제목 : 코로나 시대, 안전여행을 위한 SNS데이터 기반 감성숙소 추천서비스
- DS 파트 진행
- W2V / K-menas / TextRank / 콘텐츠 기반 필터링 알고리즘 구축



## 2021-10-20(수)

### 활동내용 : 비상교육 AI기획팀 자소서 작성



- 비상교육 AI 기획팀 자소서를 작성하였다. 경력 1년 이상부터 모집하긴 하는데 사실 어떻게 될지 모르겠다. 그렇지만 너무나도 탐나는 자리이고 경험삼아 한번 도전해보는 것도 괜찮을 것같다.
- 이제 교육이 끝나고 취업을 해야할 때이니 조금만 더 정신차리고 더 열심히 준비를 해나가면 분명 좋은 결과가 있을 것 같다!

## 2021-10-27(수)

### 활동내용 : Git Hub 꾸미기 1일차



- Git에 있는 이스터 에그 기능을 사용하면, Git 프로필을 꾸밀 수 있다고 한다. 해당 기능을 사용하여 Git 프로필 꾸미기 시작
- Readme를 만들어 해당 내용을 채워넣으면 깃 프로필로 연동이 가능하다.



## 2021- 10-28(목)

### 활동내용 : Git Hub 꾸미기 2일차



- 데일리 코딩 시간 나타내기
- Git Hub Stats 나타내기
- 프로필 표지 및 Tech Stack 나타내기 작업 완료 



## 2021-10-29(금)

### 활동내용 : Git Hub 꾸미기 3일차



- 가운데 정렬



## 2021-11-06(금)

### 활동내용 : 메가존 면접, 교원그룹 & 고려해운 자소서 작성



- 메가존 면접 
  - 직무 : 데이터분석 / 데이터 시각화
  - 일시 : 11월 14일(화) 오후 4시
  - 장소 : 메가존 건물
- 교원그룹 자소서 작성 > 90% 완료
- 고려해운 자소서 작성 > 50% 완료





## 2021-11-7(일)

### 활동내용 : 교원그룹 입사지원서 제출

 

- 고려해운 추가 지원
- 퍼플아카데이 추가 지원



## 2021-11-8(월)

### 활동내용 : 고려해운 입사지원서 제출

- 기업설명회 참석



## 2021-11-10(수)

### 활동내용 : 교원그룹 역량평가

- 일정 정리



## 2021-11-11(목)

### 활동내용 : 고려해운 서류 합격

- 고려해운 AI역량평가 준비
- 대웅그룹 지원 예정



## 2021-11-12(금)

### 활동내용 : SQLD 학습

- SQLD 시험 준비
- 빅데이터 분석기사 실기 시험 준비



## 2021-11-15(월)

### 활동내용 : 노션 이력서 및 포트폴리오 작업

- 노션 작업, 프로젝트 재정리
- 메가존 면접 준비



## 2021-11-16(화)

### 활동내용 : 메가존 면접 후기

- 1시 면접 / 지원자 3명 / 면접관 2명
- 자기소개로 면접 시작
- 질문 내용 
  - SQL은 사용해본 경험이 있는지? SQL 관련하여서 학습은 얼마나?(공통)
  - AWS와 같은 클라우드 환경을 경험해본 적 있는지?(공통)
  - R&D / 구축 / 운영 세 파트 중 어느 파트가 본인에게 잘 맞는다고 생각하는지?(공통)
  - 파이썬 이외에 다른 언어나 툴은 어떤 걸 접해 본 적이 있는지?(공통)
  - 데이터 관련 업무 경험이 있는지?(공통)
  - 해외 인턴 관련 질문 / DB 관련 / 정보윤리관련 질문(개인)
  - 풀스택 엔지니어링의 정의가 무엇인지?(개인)
  - 자바 스크립트의 비동기 처리에 대해 설명할 수 있는지?(개인)
  - 태블로 관련 질문 > 태블로를 익히고 사용하는데에 얼만큼 시간이 소요되었는지? > 학습 의지 및 학습역량에 대한 질문으로 생각된다.

- 후기 
  - 개인 질문을 수월하게 대답하지 못하였다. 분석 관련 질문보단 기본 프로그래밍 소양을 묻는 듯한 질문이 많았고, 특히 해외 인턴과 관련하여서 질문이 많은 편이었다. 프로젝트에 대한 질문이 전혀 없었다. 이유는 알 수 없지만 학원 수준의 팀 프로젝트에는 큰 관심이 없는 듯 보였다. 오히려 연구 수준의 플젝에는 질문을 많이 하여서 질문 기준이 애매하나, 자기소개의 영향도 있는듯 보였다. 신입이라서 그런지 난이도 높은 질문은 크게 나오지 않았지만 학습 역량이나 의지, 기본 IT 개념에 대해 묻는 질문이 많았기에 해당 부분에 대한 학습을 더 진행해야 할 것으로 생각된다. 자바 스크립트에 대한 기초 학습을 더 해야겠다.



## 2021-11-22(월)

### 활동내용 : 오뚜기 입사지원, SQLD 시험 완료, 빅데이터 분석기사 실기 준비



## 2021-11-23(화)

### 활동내용 : SKT 입사지원(계약직)



## 2021-11-24(수)

### 활동내용 : 케이뱅크 인턴 지원 / 노션 포트폴리오 수정



## 2021-11-30(화)

### 활동내용 : 와인 분류 연습



## 2021-12-04(토)

### 활동내용 : 빅데이터분석기사 실기시험



## 2021-12-09(목)

### 활동내용 : 노션 수정

- 2번 프로젝트 내용 추가
- 최종 확인
- 추가 입력 내용 도출



## 2021-12-10(금)

### 활동내용 : 노션에 자기계발일지 만들기 추가



## 2021-12-11(토)

### 활동내용 : 채용설명회 신청



## 2021-12-13(월)

### 활동내용 : 데이콘 직업 추천 알고리즘 경진대회 참여

- 경진대회 시작 1일차 > 작업 내용 노션에 추가 예정



## 2021-12-18(토)

### 활동내용 : GAIQ 준비



## 2021-12-19(일)

### 활동내용 : SQLD합격



## 2021-12-21(화)

### 활동내용 : 구글 애널리틱스 베이직코스 수료 완료



## 2021-12-22(수)

### 활동내용 : GAIQ & 빅분기 합격



## 2021-12-23(목)

### 활동내용 : 와인분류 노션 업로드

- 와인분류 데이터 분석 내용을 노션에 업로드하였다.



## 2021-12-24(금)

### 활동내용 : 버블콘 입사지원

- 잡코리아를 통해 지원서 작성 후 지원



## 2021-12-25(토)

### 활동내용 : 비저블 2기 지원

- 대시보드 기획서 작성
- 대시보드 설계서 작성



## 2021-12-27(월)

### 활동내용 : 퍼블리 입사지원



## 2021-12-28(화)

### 활동내용 : 트랜스코스모스 입사지원



## 2021-12-29(수)

### 활동내용 : 스포티비 입사지원



## 2021-12-30(목)

### 활동내용 : 로켓펀치 및 사람인 이력서 추가 작성



## 2021-12-31(금)

### 활동내용 : 인사이저, KT, 씨드앤 입사지원



## 2021-01-02(일)

### 활동내용 : 한국정보통신진흥협회 입사지원



## 2021-01-03(월)

### 활동내용 : 한국정보통신진흥협회 입사지원



## 2021-01-04(화)

### 활동내용 : 텍스트 분석 복습 - TF-IDF / Word2Vec



## 2021-01-05(수)

### 활동내용 : 인사이저 면접 준비 - 자소서&포트폴리오 중심으로



## 2022 - 01 - 08(토)

### 활동내용 : 면접 내용 정리



## 2022-01-09(일)

### 활동내용 : 면접 후기 정리



## 2022-01-10(월)

### 활동내용 : 뉴스데이터 텍스트 분석 실시 



## 2022-01-11(화)

## 활동내용 : 형태소 분석기 선정 / 분석 모델 설정



## 2022-01-14(금)

### 활동내용 : 2차 면접 준비



## 2022-01-15(토)

### 활동내용 : 2차 면접 준비 / M사 입사 지원



## 2022-01-16(일)

### 활동내용 : 2차 면접 준비 / PPT 제작 
