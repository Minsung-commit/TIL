# KNN Imputation

최근접 이웃을 활용한 결측치 처리 기법으로, 원칙적으론 numeric feature에 사용한다.  일반적인 사이킷 런에서 사용되는 simple imputation과 달리 변수간의 상관관계를 고려하며, 이를 통해 조금 더 신뢰도 있는 결측치 대체 값을 얻어 낼 수 있다. 



fancyimputer에 내장되어 있는 knn()을 사용하였다.

